# multipitch
<h1 align="center">Speaker-independent Multi-Pitch Tracking in Noisy and Reverberant Scenarios</h1>

This repository provides the code, pretrained models, and a semi-manually labeled test set from the study:

Y. Zhang, H. Wang, and D.L. Wang, "Speaker-independent Multi-Pitch Tracking in Noisy and Reverberant", to be submitted, 2025

## Description

Pitch tracking in multi-talker scenarios is challenging due to the difficulty of accurately estimating speech pitch in the presence of interfering speech and correctly associating each pitch with its corresponding speaker. These difficulties become more significant in noisy or reverberant environments. This study discusses key challenges in multi-pitch tracking in both clean and noisy reverberant environments, and explores several approaches to improve performance, focusing on label accuracy, training strategies, and model design.

This repository consists of two parts:

1. `src/` contains the proposed models trained under noisy and reverberant conditions, using anechoic and reverberant speech as references. More details can be found in `src/Readme.md`.

2. Libri-manual test set.

- `Libri_tt_reverb_manual_pitch.zip` comprises 140 utterances randomly selected from the LibriSpeech test-clean subset. This test set is used for evaluating pitch tracking performance for models that treat reverberant speech as reference. Each selected utterance is simulated under seven configurations described in Table A.II of [1]. The pitch is semi-manually labeled using an interactive pitch detection algorithm [1].

## References

[1] Z. Jin and D. Wang, “HMM-based multipitch tracking for noisy and reverberant speech,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, pp. 1091–1102, 2010.