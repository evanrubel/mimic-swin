import os
import PIL.Image
import pandas as pd
import PIL
from datasets import (
    GeneratorBasedBuilder,
    DatasetInfo,
    Features,
    Image,
    ClassLabel,
    Sequence,
    SplitGenerator,
    Split,
    Value,
)
import torch
import tqdm


class MCSBuilder(GeneratorBasedBuilder):
    """The MIMIC-CXR-SWIN dataset builder."""

    classes = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    def __init__(self, **kwargs):
        data_dir = kwargs["data_dir"]
        label_type = kwargs["label_type"]
        views = kwargs["views"]
        is_testing = kwargs["is_testing"]
        can_skip = kwargs["can_skip"]
        on_server = kwargs["on_server"]
        num_samples = kwargs["num_samples"]

        del kwargs["data_dir"]
        del kwargs["label_type"]
        del kwargs["views"]
        del kwargs["is_testing"]
        del kwargs["can_skip"]
        del kwargs["on_server"]
        del kwargs["num_samples"]

        super().__init__(**kwargs)

        assert len(views) >= 1, "Expected at least one view!"

        self.file_prefix = "mimic-cxr-2.0.0-"
        self.split_names = ["train", "test"]

        self.data_dir = data_dir
        self.views = views
        self.is_testing = is_testing

        # used for generating the first n samples (testing only)
        # assumes an 80/20 train/test split, and 20/80 = 0.25
        self.test_p = 0.25

        self.can_skip = can_skip
        self.on_server = on_server
        self.num_samples = num_samples

        self.metadata = self.load_metadata(label_type)  # filtered data
        self.split_to_dicom_id = self.load_splits(label_type)

    def load_metadata(
        self, label_type
    ) -> dict[str, tuple[str, str, str, torch.Tensor]]:
        """Returns a dictionary mapping DICOM ids to tuples containing
        the corresponding (subject_id, study_id, view_position, labels).

        `labels` is a length-14 binary vector of labels in the cleaned format
        with the collapsed values.

        We filter the dataset based on the labels we use!
        """

        labels_df = pd.read_csv(
            os.path.join(self.data_dir, f"{self.file_prefix}{label_type}.csv")
        )

        # change NaNs to 0
        # drop all -1.0
        cleaned_labels_df = labels_df.fillna(0).replace(-1.0, pd.NA).dropna()

        labels = []

        for _, row in cleaned_labels_df.iterrows():
            labels.append(
                torch.tensor(row.to_list()[2:], dtype=torch.float)
            )  # drop subject_id and study_id

        filtered_study_id_to_labels = dict(
            zip(
                cleaned_labels_df.study_id,
                torch.stack(labels),
            )
        )

        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, f"{self.file_prefix}metadata.csv")
        )

        # only choose the filtered metadata based on the filtered labels
        filtered_metadata_df = metadata_df[
            metadata_df["study_id"].isin(list(filtered_study_id_to_labels.keys()))
        ]

        # insert the labels
        filtered_metadata_df["labels"] = filtered_metadata_df["study_id"].map(
            filtered_study_id_to_labels
        )

        # only the desired views
        final_metadata_df = filtered_metadata_df[
            filtered_metadata_df["ViewPosition"].isin(self.views)
        ]

        return dict(
            zip(
                final_metadata_df.dicom_id,
                zip(
                    final_metadata_df.subject_id.astype(str),
                    final_metadata_df.study_id.astype(str),
                    final_metadata_df.ViewPosition.astype(str),
                    final_metadata_df.labels,
                ),
            )
        )

    def load_splits(self, label_type) -> dict[str, list[str]]:
        """Returns a dictionary mapping split to a list of the DICOM ids in that split."""
        split_df = pd.read_csv(
            os.path.join(
                f"metadata/{self.file_prefix}split-{label_type}-{'-'.join(self.views)}.csv"
            )
        )

        dicom_id_to_split = dict(zip(split_df.dicom_id, split_df.split))

        metadata = list(self.metadata.keys())

        train, test = [], []

        for dicom_id in metadata:
            if dicom_id_to_split[dicom_id] == "train":
                train.append(dicom_id)
            elif dicom_id_to_split[dicom_id] == "test":
                test.append(dicom_id)
            else:
                raise NotImplementedError

        return {
            "train": train[: (self.num_samples if self.is_testing else len(train))],
            "test": test[
                : (
                    int(self.num_samples * self.test_p)
                    if self.is_testing
                    else len(test)
                )
            ],
        }

    def _info(self):
        return DatasetInfo(
            description="The MIMIC-CXR-SWIN dataset.",
            citation="Based on https://doi.org/10.13026/th9c-ae10",
            homepage="Based on https://physionet.org/content/mimic-cxr-jpg/2.1.0/",
            license="https://physionet.org/content/mimic-cxr-jpg/2.1.0/LICENSE.txt",
            features=Features(
                {
                    "dicom-id": Value("string"),
                    "image": Image(),
                    "labels": Sequence(
                        feature=ClassLabel(num_classes=2), length=len(self.classes)
                    ),
                }
            ),
        )

    def get_dicom_path(self, dicom_id):
        """Given a DICOM id, return the path to the corresponding DICOM based on the hierarchical form of the data."""

        subject_id, study_id, _, _ = self.metadata[dicom_id]  # pXXXX..., sYYYY...

        # based on directory hierarchy
        return os.path.join(
            self.data_dir,
            "files" if self.on_server else "data/files",
            f"p{subject_id[:2]}",
            f"p{subject_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )

    def get_dicom_image(self, dicom_id):
        """Given a DICOM id, return the corresponding DICOM image."""

        image = PIL.Image.open(self.get_dicom_path(dicom_id))
        image.close()
        return image

    def get_labels(self, dicom_id):
        """
        Given a DICOM id, return a length-14 binary tensor of labels.

        This is in the cleaned format with the collapsed values.

        If it is a tensor of all zeros, then we insert a single 1 in the No Findings index (index 8).
        """

        labels = self.metadata[dicom_id][3]

        if torch.count_nonzero(labels) == 0:  # all zeros
            only_no_findings = [0 for _ in range(len(self.classes))]
            only_no_findings[8] = 1

            return torch.tensor(only_no_findings, dtype=torch.float)

        else:
            return labels

    def _compile_data(self, split):
        """Returns a list of (dicom_path, dicom_id, image, labels) from the split."""

        data = []

        for dicom_id in tqdm.tqdm(self.split_to_dicom_id[split]):
            try:
                data.append(
                    (
                        self.get_dicom_path(dicom_id),
                        dicom_id,
                        self.get_dicom_image(dicom_id),
                        self.get_labels(dicom_id),
                    )
                )
            except FileNotFoundError:
                if self.is_testing:
                    print(f"Skipping {dicom_id}...")

                if not self.can_skip:  # if we can't skip it, raise an error
                    raise FileNotFoundError

        return {
            "split": split,
            "data": data,
        }

    def _split_generators(self, dl_manager):
        """Assumes files are already downloaded."""

        split_name_to_hf_name = {
            "train": Split.TRAIN,
            "test": Split.TEST,
        }

        return [
            SplitGenerator(
                name=split_name_to_hf_name[split_name],
                # These kwargs will be passed to _generate_examples below
                gen_kwargs=self._compile_data(split_name),
            )
            for split_name in self.split_names
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, **kwargs):
        """Generate images and labels for a given split."""

        # (filtered) list of (dicom_path, dicom_id, image, labels) from that split
        for dicom_path, dicom_id, image, labels in kwargs["data"]:
            yield dicom_path, {
                "dicom-id": dicom_id,
                "image": image,
                "labels": labels,
            }
