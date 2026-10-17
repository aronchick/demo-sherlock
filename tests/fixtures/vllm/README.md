# Benchmark photograph

`shuttle_crop.jpg` is a 110 × 240 crop of the model spacecraft beside
astronaut Eileen Collins in a real NASA photograph. The crop excludes the
person. It exercises image decoding, image-token prefill and the imported
Analyst/Verifier prompts with an object-only question and claim.

Source: [scikit-image v0.24.0 astronaut.png](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png).
[Provenance and license](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut):
NASA Great Images database; no known copyright restrictions, public domain.

Preparation: `astronaut_crop.jpg` used source pixels
`image[40:424, 96:480]`, JPEG quality 85. The current fixture takes
`image[15:255, 270:380]` from that JPEG, again at quality 85. The older
portrait crop is retained only to reproduce the historical stub results;
the benchmark uses `shuttle_crop.jpg` by default.
Use a representative camera object crop via `--image` for the final run.
Each result records source and transmitted JPEG SHA-256 digests and dimensions.
