# Non-sensitive dataset metadata (white-headed langur recordings)

This file documents the recording protocols for the white-headed langur
(*Trachypithecus leucocephalus*) dataset analysed in the paper.

**Withheld by design:** GPS coordinates of the autonomous recording units (ARUs)
and precise deployment dates/locations are withheld because the species is IUCN
Critically Endangered and its location information is sensitive; they are
available from the corresponding author on reasonable request. A deployment
overview map without coordinates is provided in Supplementary Information
(Figure S4).

## Manual recordings (training data and reference construction)

- **Period:** April 2020 – August 2023 (41 months), covering both the wet
  (April–September) and dry (October–March) seasons.
- **Effort:** one focal group per day on a rotating basis (15–25 observation
  days per month); daily observation windows 05:30–20:00 (wet season) and
  06:00–19:00 (dry season).
- **Targets:** the three long-distance call types of adult males
  (snort, roar, wahoo).
- **Equipment:** Zoom F6 digital recorder with a Rode NTG8 or Sennheiser ME66
  directional microphone; 44.1 kHz sampling rate, 16-bit depth, WAV format.
- **Event logging:** timestamp, caller identity, and behavioural context were
  logged for each vocal event.

## Passive acoustic monitoring (PAM) recordings

- **Units:** 12 solar-powered autonomous recording units (Lindon Ecology,
  model LAVian-01211); mono MP3, 32 kHz, 16-bit.
- **Deployment protocol:** units mounted on metal poles 3–4 m above ground;
  minimum spacing 200 m between units; effective recording radius ≈100 m in
  dense forest.
- **Schedule:** 12 h per unit per day, aligned with the species' peak
  vocalization periods (05:00–12:00 and 15:00–20:00); contiguous,
  non-overlapping 12-second audio files.
- **Deployment phases:** September–December 2023 within the home ranges of
  three habituated groups (four ARUs per group); January–March 2024 within the
  range of one unhabituated group.
- **Maintenance:** equipment inspected every 2–3 weeks.

## Acoustic measurements

- Call amplitude at ≈80 m: 48.2–54.4 dB; concurrent ambient noise at the
  caller's location: 38.2–41.9 dB (SMART SENSOR AR814 sound level meter;
  SNDWAY SW-600A rangefinder).

## Example audio

The repository-level `data/` directory contains a small set of paired
noisy/denoised example clips for the three studied taxa (white-headed langur,
anuran, avian). These examples are provided to demonstrate the denoising
pipeline; they are not part of the evaluation partitions.
