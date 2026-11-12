# Video data

No real conveyor footage is available yet. Keep raw videos under `data/raw/` locally;
the directory is ignored by Git. Record an empty belt, oranges alone, apples alone,
both fruits together, and cases with close spacing or brief occlusion. Split train,
validation, and test by recording session so nearly identical frames do not leak
between splits. Record each video's expected orange and apple crossing counts.

## Public pretraining data

`fruit-counter prepare-coco` builds `data/datasets/coco-fruit/` (YOLO format) from
COCO 2017 images containing oranges or apples. Annotations are CC BY 4.0. Each image
keeps its own Flickr license, which is listed in `licenses.csv`. Both `data/datasets/`
and `data/cache/` are ignored by Git.

The command does not create any conveyor-belt images. Label frames from the real line
(for example in CVAT, exported as YOLO 1.1) with class 0 = orange and 1 = apple. Fine-tune
on those frames, and evaluate counting only on held-out recording sessions.
