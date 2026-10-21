# Dataset attribution

## Growth-stage object detection

- Source project: **Bamboo Detection and Classification**, version 6.
- Publisher/workspace: **bamboo-6gkce**, hosted on Roboflow Universe.
- Source: https://universe.roboflow.com/bamboo-6gkce/bamboo-detection-and-classification/dataset/6
- License recorded in the downloaded source metadata: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- License text: https://creativecommons.org/licenses/by/4.0/
- Classes: `matured`, `old`, `young`.
- Changes: the BambooX team cleaned and screened the dataset for duplicate/similar images; this website distributes the resulting `growth_stage_clean` images and bounding-box annotations. Image filenames and annotation coordinates are preserved. Quarantine folders are not included.

The dataset is research material. Its class names are dataset-defined categories, not an independent determination of a plant's biological age or structural safety.

## Species model

The team-trained MobileNetV3-Small checkpoint uses nine BambooBD species classes. The species training image dataset is not redistributed by this website. Consult the original dataset source for its licensing and citation requirements before redistributing those training images.
