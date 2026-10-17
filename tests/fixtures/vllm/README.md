# Benchmark photograph

`astronaut_crop.jpg` is a 384 × 384 crop of a real NASA photograph of
astronaut Eileen Collins. It exercises image decoding, image-token prefill
and the Analyst/Verifier prompts. It is not a synthetic image or a claim
about Sherlock's field accuracy.

Source: [scikit-image v0.24.0 astronaut.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png).
[Provenance and license](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut):
NASA Great Images database; no known copyright restrictions, public domain.

Preparation: source pixels `image[40:424, 96:480]`, JPEG quality 85.
The benchmark can use a representative camera crop instead via `--image`.
Each result records source and transmitted JPEG SHA-256 digests and dimensions.
