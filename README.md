
## SIFT-Based Detection of an Almond Bag in Cluttered Real-World Scenes

**Course:** MSML640  
**Semester:** Spring 2026  
**Group:** Group 8  
**Team Members:** Haider Khan, Batamgalan Enkhtaivan

---

## Project Overview

This project explores how well a classical computer vision method, **SIFT (Scale-Invariant Feature Transform)**, can detect a target almond bag in real-world scene images.

Instead of using deep learning, we focused on a traditional feature-based approach. The goal was to test whether SIFT could still find the target object when the scene becomes more difficult, such as under:

- cluttered backgrounds  
- different camera angles  
- scale changes  
- low lighting  
- partial occlusion  
- visually similar distractor objects

This project follows the course goal of studying how classical computer vision behaves in unpredictable “in-the-wild” settings.

---

## Why This Project Fits the Assignment

The project is built around **SIFT**, which is one of the allowed core techniques in the rubric.  
It uses **real-world image data** rather than synthetic toy examples.  
It also emphasizes:

- preprocessing
- parameter tuning
- failure analysis
- improvement through iteration

That means the project is not just about getting detections, but about understanding **when the algorithm succeeds, when it fails, and why**.

---

## Final Application

Our final application detects whether a scene image contains the **target almond bag**.

The system:

1. loads clean reference images of the target bag
2. extracts SIFT keypoints and descriptors from both reference and scene images
3. matches local features between images
4. filters weak matches using the Lowe ratio test
5. estimates a homography using RANSAC
6. decides whether the bag was detected based on:
   - number of good matches
   - number of inliers
   - whether the projected polygon is reasonable

The output includes:

- a **match visualization** showing feature correspondences
- an **overlay image** showing the detected object boundary
- a **CSV log** with per-image evaluation results

---

## Repository Structure

```text
.
├── data/
│   ├── reference_v2/              # final reference images
│   ├── scene_positive_v2/         # final positive scenes containing target
│   ├── scene_negative_v2/         # final negative scenes without target
│   ├── holdout_positive_v2/       # holdout positives for additional testing
│   ├── holdout_negative_v2/       # holdout negatives for additional testing
│   └── backup_before_curated_v2/  # earlier backup of data split
│
├── docs/                          # place proposal PDF and any extra documentation here
├── slides/                        # place final presentation slides here
│
├── results/
│   ├── logs/                      # CSV evaluation logs
│   ├── matches_v2/                # feature match visualizations
│   ├── figures_v2/                # detection overlays
│   ├── matches_holdout_v2/        # holdout match visualizations
│   └── figures_holdout_v2/        # holdout detection overlays
│
├── src/
│   ├── evaluate.py
│   ├── run_demo.py
│   ├── sift_detector.py
│   ├── streamlitapp.py
│   └── utils.py
│
├── README.md
└── requirements.txt
````

---

## Dataset

The dataset contains three main image types:

### 1. Reference images

These are clean views of the target almond bag used as the object template.

Example folder:

* `data/reference_v2/`

### 2. Positive scene images

These images contain the target bag, but under more realistic conditions such as clutter, viewpoint changes, and harder backgrounds.

Example folder:

* `data/scene_positive_v2/`

### 3. Negative scene images

These images do **not** contain the target bag.
They are used to test whether the detector avoids false positives.

Example folder:

* `data/scene_negative_v2/`

### 4. Holdout sets

We also kept separate holdout positive and negative sets for additional evaluation beyond the main split.

Example folders:

* `data/holdout_positive_v2/`
* `data/holdout_negative_v2/`

### Data source style

The project proposal planned a mix of practical, real-world style images, including cluttered scenes such as shelves, jars, bowls, and similar nut-related settings. The purpose was to test the algorithm under realistic interference instead of ideal lab conditions.

---

## Method

### Step 1: Image loading and resizing

All images are loaded in color and resized so the maximum dimension stays manageable for processing.

### Step 2: Preprocessing

Before extracting SIFT features, the system converts images to grayscale.
Optional preprocessing includes:

* CLAHE contrast enhancement
* Gaussian blur

These were used experimentally to test whether feature detection became more stable in harder scenes.

### Step 3: SIFT feature extraction

SIFT keypoints and descriptors are extracted from both the reference and scene images.

### Step 4: Descriptor matching

A brute-force matcher is used to compare descriptors between reference and scene images.

### Step 5: Lowe ratio test

Weak or ambiguous matches are filtered out using the Lowe ratio test.

### Step 6: Homography estimation

If enough matches remain, the system uses RANSAC to estimate a homography between the reference and scene image.

### Step 7: Detection decision

The detector predicts that the almond bag is present only if:

* enough good matches are found
* enough inliers survive RANSAC
* the projected polygon is geometrically reasonable

This makes the detector more conservative and helps reduce false detections.

---

## Final V2 Configuration

The final version mainly used the following settings:

* `ratio_test = 0.55`
* `min_good_matches = 12`
* `min_inliers = 12`
* `max_dim = 1000`
* CLAHE disabled by default for final v2
* Gaussian blur disabled by default for final v2

These values were selected after iterative testing to make the detector stricter and more stable in cluttered scenes.

---

## How to Set Up

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Important: SIFT requires an OpenCV build that includes contrib modules.
> If needed, make sure `opencv-contrib-python` is included in `requirements.txt`.

---

## How to Run the Project

### Option 1: Run full evaluation on the final v2 dataset

```bash
python src/evaluate.py
```

This uses the default final v2 folders:

* `data/reference_v2`
* `data/scene_positive_v2`
* `data/scene_negative_v2`

and saves outputs to:

* `results/logs/bag_eval_v2.csv`
* `results/matches_v2/`
* `results/figures_v2/`

---

### Option 2: Run on a single image or folder

Run a demo on one image:

```bash
python src/run_demo.py --scene data/scene_positive_v2/Scene_Positive1.jpeg
```

Run a demo on a folder:

```bash
python src/run_demo.py --scene-dir data/scene_positive_v2
```

Outputs are saved to:

* `results/demo_v2/` if specified
* or the chosen output directory

Example with a custom output folder:

```bash
python src/run_demo.py --scene-dir data/scene_positive_v2 --output-dir results/demo_v2
```

---

### Option 3: Run the Streamlit app

```bash
streamlit run src/streamlitapp.py
```

The app allows the user to:

* upload one or more scene images
* adjust thresholds from the sidebar
* run the detector interactively
* inspect overlay and match visualizations
* download a summary CSV

This interface is included to make the project easier to demonstrate, especially for a non-CS audience.

---

## Main Files

### `src/sift_detector.py`

Contains the main SIFT detector logic, including:

* SIFT configuration
* feature extraction
* ratio test
* homography estimation
* detection scoring
* best-reference selection

### `src/evaluate.py`

Runs evaluation over positive and negative scene folders and writes a CSV summary.

### `src/run_demo.py`

Runs the detector on a single image or a directory of images.

### `src/streamlitapp.py`

Interactive web demo for uploading images and viewing results.

### `src/utils.py`

Helper functions for:

* image loading
* preprocessing
* resizing
* polygon validation
* visualization
* image saving

---

## Results Summary

This project is not meant to show a perfect detector.
Instead, it is meant to show how a classical feature-based method performs under realistic visual interference.

### What worked well

The detector worked best when:

* the almond bag was visible from the front or near-front view
* the printed texture on the package was clear
* lighting was reasonable
* the object was not too blurred or heavily occluded
* the target occupied enough image area for stable keypoint extraction

### What became difficult

The detector became less reliable when:

* the bag was heavily rotated
* only a small portion of the target was visible
* blur reduced local texture detail
* glare or poor lighting weakened keypoints
* similar nut-related packaging or clutter created confusing local features

### Why the improvements mattered

The project improved through iteration by making the detector stricter and more robust:

* cleaner v2 reference images were used
* thresholds were tuned
* match filtering was made stricter
* polygon sanity checks were added
* holdout testing was added to better judge generalization

These changes helped reduce weak detections and made the final pipeline more believable.

---

## Technical Reflection

A major lesson from this project is that **classical computer vision can still work well in practical settings**, but only when its assumptions are respected.

SIFT is useful because it can handle some amount of scale and rotation change, but it is still sensitive to:

* low texture
* severe blur
* strong occlusion
* confusing repeated patterns
* poor-quality reference images

This project shows that performance does not only depend on the algorithm itself. It also depends heavily on:

* data quality
* reference image choice
* preprocessing
* threshold tuning
* rejection logic

In other words, a simple pipeline can become much stronger when the failure modes are studied carefully.

---

## Limitations

This project has several limitations:

* It focuses on one target product class
* It does not use deep learning or large-scale object detection
* It may struggle on extreme viewpoint changes
* It may fail when the target is only partially visible
* It may confuse visually similar packaging in some difficult scenes

These limitations are expected and are part of the technical analysis.


---
AI and External Sources Disclosure

 We used external sources only for allowed support tasks such as writing cleanup,streamlit app etc, and presentation wording. The core project idea, dataset collection, SIFT pipeline design, implementation, debugging, parameter tuning, and evaluation logic were completed by the team using several different documentation and lecture slides. 

Links that we used as reference;


https://github.com/opencv/opencv_contrib
https://github.com/opencv/opencv
https://docs.opencv.org/4.x/d4/d5d/group__features2d__draw.html
https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
https://docs.python.org/3/library/tempfile.html
https://github.com/daniel1896/Feature-Matching-and-Homography
https://github.com/jvirico/sift-feature-matching
https://github.com/mtszkw/matching
https://github.com/dalgu90/opencv-tutorial/blob/master/4_homography.ipynb
https://github.com/tobybreckon/python-examples-cv/blob/master/sift_detection.py
https://github.com/streamlit/streamlit
https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
https://docs.streamlit.io/develop/api-reference/data/st.dataframe
https://docs.streamlit.io/develop/api-reference/widgets/st.download_button


Some promopts that we used for the streamlit not limitted too we looked at the code and shared some code snippit for debug as well for the streamlit app integration . The actual prompt that we used are in doc files. 

