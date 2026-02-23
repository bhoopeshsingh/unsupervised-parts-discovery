# How the System Works — Explained Simply

> *Written so that anyone — including a Class 5 student — can understand exactly what this system does and why it is clever.*

---

## The Real-World Problem We Are Solving

Imagine you are a doctor. Every day, hundreds of patients send you their X-ray or scan pictures. You have to look at each one and say: **"This person is sick"** or **"This person is healthy"**.

Now — what if there were so many pictures that you could not look at all of them yourself? You would want a **computer helper** to do it for you.

The problem with most computer helpers today is this:

> The computer says **"sick"** or **"healthy"** — but it cannot tell you **WHY**.

It cannot say: *"I think this person is sick because I saw a shadow in the upper part of the lung, and the texture there looks unusual."*

It just says **"sick"** and gives no reason.

**A doctor cannot trust that.** If the computer is wrong, the doctor has no way to catch the mistake.

---

## What Our System Does Differently

Our system does something special. It does not just say sick or healthy.

It says:

> **"This patient is SICK (82% confident) — because I found these specific patterns:"**
> - Pattern A (shadow near edge): 16% of the scan shows this
> - Pattern B (irregular core): 4% of the scan shows this
> - Pattern C (unusual texture): 12% of the scan shows this
> - *[colour-coded map showing exactly WHERE on the scan each pattern appears]*

Now the doctor can **look at the map**, **check the reasons**, and **agree or disagree**.

This is called **Explainable AI** — the computer explains itself like a student explaining their answer to a teacher.

---

## The Story of Our System

### Think of it like a Detective School

Imagine you run a school that trains detectives to identify cats in photos.

But there is a twist — you cannot just tell the computer "a cat has pointy ears, whiskers, and fur." The computer has to **figure out what a cat looks like on its own**, just by looking at thousands of cat photos.

And then — when it sees a new photo — it must say:
1. **Is this a cat? Yes or no.**
2. **Which parts of this photo made you think that?**
3. **Show me exactly where those parts are.**

That is exactly what our system does. Let us walk through each step.

---

## Step 1 — The Very Smart Artist (DINO)

**What happens:** The computer looks at each image and describes every tiny part of it.

**The simple story:**

Imagine a very talented artist who has spent 10 years studying millions of photographs — cats, dogs, birds, cars, mountains, everything. This artist has an incredible memory. When you show him any photo, he can describe every small patch of it in great detail.

We use an AI called **DINO** as our artist. DINO was trained by Facebook on 1.2 million photos. It knows what visual patterns look like — textures, shapes, edges, colours in context.

**What DINO actually does:**

When you give DINO a photo, it cuts the photo into a **grid of tiny squares** — like cutting a pizza into 28 × 28 = **784 small pieces** (called "patches").

```
[  ][  ][  ][  ]...[  ]   ← row 1  (28 patches)
[  ][  ][  ][  ]...[  ]   ← row 2
...
[  ][  ][  ][  ]...[  ]   ← row 28

Total: 784 tiny patches per image
Each patch = 8 × 8 pixels
```

For each of those 784 patches, DINO produces a **description** — a list of 384 numbers that captures everything meaningful about that patch (its texture, its colour, what surrounds it, what shape it is part of).

```
Patch at row 5, col 12  →  [0.23, -0.41, 0.87, 0.12, ... ]  (384 numbers)
Patch at row 5, col 13  →  [0.21, -0.39, 0.88, 0.11, ... ]  (384 numbers)
... and so on for all 784 patches
```

**Why 384 numbers?** Think of it like describing a person. You might use height, weight, hair colour, eye colour, skin tone... Our system uses 384 such "measurement dimensions" for each patch, capturing things that are impossible to name but that the AI learned are important.

**The key insight:** Two patches that look similar (e.g. two different cat ears from two different photos) will have very similar 384-number descriptions. Patches that look different (ear vs fur vs eye) will have very different descriptions. DINO naturally groups similar-looking patches close together in this 384-number space.

---

## Step 2 — Ignoring the Messy Bedroom (Foreground Masking)

**What happens:** The computer throws away patches that belong to the background.

**The simple story:**

When you take a photo of your pet cat, there is often a sofa, a wall, or a floor in the background. If you try to learn "what makes a cat look like a cat", you do not want to study the sofa — you want to study only the cat.

DINO has a clever built-in ability: **its attention** naturally focuses on the most important parts of the image. Think of it like a student's eyes in a classroom — naturally drawn to the teacher, not to the window.

**How it works internally:**

DINO has something called a **CLS token** — a special "summary" piece that DINO uses to represent the whole image. This summary token pays attention to the patches it finds most important. We measure how much attention the CLS token pays to each of the 784 patches.

```
Attention of CLS token to each patch:
  Patch at the cat's eye  → attention = 0.87  (HIGH — very important)
  Patch at the cat's ear  → attention = 0.81  (HIGH)
  Patch at the background → attention = 0.12  (LOW — ignore this)
  Patch at the floor      → attention = 0.08  (LOW — ignore this)
```

We then set a **threshold** (currently 60th percentile): keep only the top 40% most-attended patches. The rest are thrown away.

**Result:** Instead of 784 patches per image, we keep about **300 patches** — only the ones on the cat's body. Background patches are discarded before any learning begins.

**Medical meaning:** In a chest X-ray, we keep only the patches on the lung area and discard patches showing the scanner table, the patient's clothing, or the image borders.

---

## Step 3 — Sorting the Giant Lego Pile (Clustering)

**What happens:** The computer groups all patches from all images into groups of patches that look similar.

**The simple story:**

Imagine you have a giant box containing **one million tiny Lego pieces** collected from thousands of photos. Your job is to sort them into groups — all the similar-looking pieces go together.

But no one tells you what the groups should be. You have to figure it out yourself just by looking.

You might end up with:
- Box 1: pieces that look like pointy triangles at the top of images → *cat ears!*
- Box 2: pieces that look like small bright round shapes → *cat eyes!*
- Box 3: pieces that look like soft, fluffy texture → *cat fur!*
- Box 4: pieces that look like flat grey → *floor/background — useless*

This is exactly what our clustering step does. The algorithm is called **GMM (Gaussian Mixture Model)**.

**Why GMM and not the simpler KMeans?**

Imagine sorting Lego pieces by size. Cat fur covers most of the cat's body (many patches per image), but cat eyes are tiny (very few patches per image). A simple sorting method (KMeans) would try to make all groups the same size — it would split the fur into many tiny groups and lump the eyes together with other things. GMM is smarter: **it handles groups of different sizes**, which is exactly what we need for body parts (eyes are rare, fur is common).

**The three tricks we use before sorting:**

**Trick 1 — Simplify the descriptions (PCA: 384 → 64 numbers)**
Instead of comparing patches using all 384 numbers, we first find the 64 most important directions (like finding the 64 most informative questions to ask about a patch). This makes clustering faster and removes noise. It still captures 85-90% of all meaningful information.

**Trick 2 — Add the patch's location (Spatial Features)**
Before sorting, we add two extra numbers to each patch's description: its row position and column position on the image (0 to 1 scale). This is scaled to 40% of the importance of the visual features.

Why? Because **cat ears are always near the top of the image**, and cat paws are always near the bottom. If we ignore position, the computer might put a top-left white patch and a bottom-right white patch in the same group even though one is an ear and one is a paw. Adding position prevents this.

**Trick 3 — Scale up (150,000 samples for fitting)**
We have over 1 million patches total — too many to sort all at once. So we take a random sample of 150,000 patches to **learn the groups**, then we **assign** all 1 million patches to those groups. (Like learning what the Lego types are by studying a sample box, then sorting the full box.)

**What comes out:**

Each of the ~1 million patches is now labelled: "Cluster 0", "Cluster 1", ..., "Cluster 11". These are 12 boxes. We do not yet know what each box represents — that comes in the next step.

---

## Step 4 — Teaching the Artist to See Better (Fine-Tuning)

**What happens:** We improve the DINO artist's descriptions using the groups we just discovered.

**The simple story:**

After Step 3, we have rough groups of patches. Now we use those groups to **teach DINO to describe patches more distinctly**.

Imagine your sorting boxes from Step 3 are a bit messy — some ear patches ended up in the fur box and vice versa. You want DINO to give ear patches and fur patches descriptions that are more different from each other, so the sorting becomes cleaner.

The teaching uses a simple rule called the **Semantic Consistency Loss**:

> **Rule 1:** If two patches are in the same box (same group) → make their 384-number descriptions more similar to each other. Pull them together.
>
> **Rule 2:** If two patches are in different boxes → make their descriptions more different from each other. Push them apart.

```
Patch A (cat ear) + Patch B (cat ear) → same group → pull descriptions closer
Patch A (cat ear) + Patch C (cat fur) → different groups → push descriptions apart
```

We only update the last 2 layers of the DINO model (out of 12 total). This is like only re-training the final polish of the artist — we keep all the deep understanding he already has, but sharpen the last step of his description process.

**After fine-tuning:** The 12 groups are much cleaner. Cat ears consistently group together, cat eyes consistently group together. The sorting job in Step 3 becomes much more reliable.

**This is done as a loop:**
```
Step 1+2: Extract features → Step 3: Cluster → Step 4: Fine-tune →
→ Step 1+2 again (with improved DINO) → Step 3 again (cleaner clusters)
```

---

## Step 5 — The Expert Gives Names (Human Labeling)

**What happens:** A human expert looks at each of the 12 boxes and gives them a name.

**The simple story:**

After the computer sorted the Lego pieces into 12 boxes, it does not know what the boxes contain. It just knows "patches in Box 3 look similar to each other."

Now you — the expert — open each box, look at a sample of the pieces, and say:

- Box 0 → *"These all look like cat ears"* → label: **cat_ear**
- Box 1 → *"These are cat eyes"* → label: **cat_eye**
- Box 2 → *"This is light-coloured fur"* → label: **light_fur**
- Box 3 → *"This is the cat's face area"* → label: **cat_face**
- Box 7 → *"This is just floor/background — not useful"* → **EXCLUDE**

This is done through a **Streamlit web app** (our labeling tool). For each cluster, you see:

1. **9 example patches** from that cluster (tiny 8×8 pixel squares, zoomed in)
2. The **full original image** with a red box showing where that patch came from
3. A **colour-coded map** of the whole image showing all 784 patch assignments

You type a label, mark whether to include or exclude, rate your confidence (1-3), and save.

**This is the most important step** — it is where human expert knowledge enters the system. The computer found the structure; the human gives it meaning.

**In medicine:** A radiologist would look at each box and say:
- Box 0 → *"These are patches showing irregular lesion margins"*
- Box 1 → *"These patches show the lesion's dark core"*
- Box 4 → *"These are normal healthy lung tissue patches"*
- Box 9 → *"This is just scanner background noise — exclude"*

---

## Step 6 — Creating a Recipe for Each Part (Concept Vectors)

**What happens:** For each named group, the computer calculates the "average description" of all patches in that group.

**The simple story:**

Now that you know Box 0 = cat_ear, the computer collects all the 384-number descriptions from every ear patch in every image, and calculates the **average**.

This average is called a **Concept Vector** — the "typical fingerprint" of a cat ear.

```
All cat_ear patches from all images:
  Patch 1: [0.23, -0.41, 0.87, ...]
  Patch 2: [0.21, -0.39, 0.85, ...]
  Patch 3: [0.25, -0.43, 0.89, ...]
  ...10,000 patches...
  Average: [0.22, -0.41, 0.87, ...]  ← This is the cat_ear Concept Vector
```

Think of it as: if you mixed all ear patches together to create the "perfect average ear", the Concept Vector is its description.

**Why this matters:** The Concept Vector is the **definition** of what a cat ear looks like in the computer's language. It is the reference point we use to measure new images.

---

## Step 7 — Counting How Much of Each Part Is in a New Image (Concept Scoring)

**What happens:** For any new image, we measure how much of each named concept it contains.

**The simple story:**

Imagine you get a new photo and you want to answer: *"How much of this photo looks like cat ears? How much looks like cat eyes? How much like cat fur?"*

**The clever trick:** We use the already-fitted GMM (the sorting machine from Step 3) to assign each patch in the new image to one of the 12 groups. Then we simply **count proportions**:

```
New image has 300 foreground patches (after masking):
  48 patches → assigned to "cat_ear" cluster     → cat_ear score = 48/300 = 0.16
  13 patches → assigned to "cat_eye" cluster     → cat_eye score = 13/300 = 0.04
   2 patches → assigned to "light_fur" cluster   → light_fur score = 2/300 = 0.01
  51 patches → assigned to "cat_face" cluster    → cat_face score = 51/300 = 0.17
```

**Why this works for disease detection:**

If you show a photo of a bird, its patches (feathers, beak, sky) were never in the cat training data. When the GMM tries to assign them to one of the 12 cat-trained clusters, they scatter randomly across all clusters — no single cat concept gets a high proportion.

```
Bird image (300 patches):
  cat_ear score = 0.000   (zero patches look like cat ears)
  cat_eye score = 0.000   (zero patches look like cat eyes)
  cat_face score = 0.001  (almost nothing)
  light_fur score = 0.008 (very little overlap)
```

Compare this to a real cat image:
```
Cat image (300 patches):
  cat_ear score  = 0.161
  cat_eye score  = 0.045
  cat_face score = 0.169
  light_fur score = 0.006
```

The scores tell a completely different story. This is the power of the cluster proportion approach.

**Earlier mistake we fixed:** We originally tried measuring similarity using a mathematical formula (cosine similarity). But that gave scores of 0.4 to 0.8 for everything — cats, birds, airplanes, random images. The computer could not tell anything apart. The cluster proportion method gives true zero for non-cat images and meaningful scores for cat images.

---

## Step 8 — The Bouncer Who Only Knows Club Members (One-Class Classifier)

**What happens:** We train a classifier using only cat (positive/disease) examples, then use it to identify non-cats.

**The simple story:**

Imagine a nightclub bouncer who has worked the door for 5 years. Every night, he has seen hundreds of club members walk in. He knows exactly what club members look like — how they dress, how they walk, what they carry.

One night, a stranger tries to get in. The bouncer looks at them and says: *"You do not look like any of our regular members. You are not coming in."*

The bouncer never studied what non-members look like. He only needed to know the members deeply.

**This is the OneClassSVM.** It is trained only on cat concept score profiles. It learns the **shape of the cat space** — what combinations of concept scores are typical for cats.

```
Training data (cat images only):
  Cat 1: [cat_ear=0.16, cat_eye=0.04, light_fur=0.01, cat_face=0.17]
  Cat 2: [cat_ear=0.14, cat_eye=0.06, light_fur=0.02, cat_face=0.15]
  Cat 3: [cat_ear=0.18, cat_eye=0.03, light_fur=0.01, cat_face=0.19]
  ...3,738 cats...

The SVM draws a boundary around this cluster of points.
Anything inside → CAT
Anything outside → NOT CAT
```

**The nu parameter (= 0.05):** This means "assume 5% of my training data might be mislabeled or weird." A small nu = tighter boundary. We use 0.05 so the boundary is tight — a bird must look very much like a cat to get in.

**Why not train it with both cat and non-cat examples?**

In real medicine:
- Positive examples (disease scans) are rare and expensive to label — a radiologist must confirm each one
- Negative examples (healthy scans) are easy to collect, but you would need examples of every possible non-disease (bird images, airplane images, random noise...) — that is infinite
- In the future, you may encounter a new disease variant not in your training set — a binary classifier would not know what to do

The one-class approach mirrors real clinical practice: *"I know exactly what the disease looks like. Flag anything that does not match."*

**How it scores a new image:**

The SVM gives each image a **score**:
- Score > 0 → inside the cat boundary → **CAT** (positive/disease)
- Score < 0 → outside the cat boundary → **NOT CAT** (negative/healthy)
- How far from 0 = confidence (far inside = very confident cat; far outside = very confident not cat)

---

## Step 9 — Writing the Report Card (Explanation)

**What happens:** For each prediction, the system generates a 3-panel image showing what it found and why.

**The simple story:**

Imagine a student who just completed an exam. The teacher does not just give a grade (A or F). The teacher shows:

1. **The question paper** — the original image
2. **The answer sheet with highlighted parts** — which parts of the image correspond to which concept
3. **The score breakdown** — which concepts helped, which ones were weak

Our system produces exactly this.

---

**Panel 1: The Original Image**
Just the original photo, unchanged.

---

**Panel 2: The Semantic Part Map**

The computer assigns each of the 784 patches to its closest concept (by comparing its 384-number description to each concept vector). It then colours each patch with a colour for that concept:

```
cat_ear patches  → 🔴 Red
cat_eye patches  → 🔵 Blue
light_fur patches → 🟢 Green
cat_face patches  → 🟡 Yellow
background        → dimmed (low attention = half transparent)
```

The 28×28 grid of colours is then stretched to cover the full 224×224 image. Background patches (from Step 2's masking) are made semi-transparent so they visually fade away.

**Result:** You can see exactly WHICH REGION of the image corresponds to which concept — and whether those regions make biological sense (ears at the top, eyes in the middle, etc.)

---

**Panel 3: The Contribution Bar Chart**

Each concept gets a bar showing whether it pushed the prediction toward "positive" or "negative":

```
cat_face  ████████████████  +0.34  (strong positive signal)
cat_ear   ███████████████   +0.28  (strong positive signal)
cat_eye   ████              +0.08  (mild positive signal)
light_fur █                 +0.01  (weak signal)
```

The bar length = (concept score) × (how much this concept deviates from a typical cat's average score).

---

**What this looks like for a CAT image:**

```
Prediction: CAT (82% confident)

Concept Activations:
  cat_face   0.169  → strong signal, many face patches detected
  cat_ear    0.161  → strong signal, ears clearly visible
  cat_eye    0.045  → present but weaker
  light_fur  0.006  → very little light fur in this image

Evidence: cat_face, cat_ear
```

**What this looks like for a BIRD image:**

```
Prediction: NOT CAT (100% confident)

Concept Activations:
  cat_face   0.000  → zero face patches detected
  cat_ear    0.000  → zero ear patches detected
  cat_eye    0.000  → zero eye patches detected
  light_fur  0.008  → tiny overlap (feathers slightly similar to light fur)

Evidence: none — no cat-like features detected
```

---

## The Full Journey — All Steps Together

Here is the complete journey of one cat photo through the system:

```
📸 PHOTO: "cat_001.jpg"
          │
          ▼
🎨 DINO cuts it into 784 tiny patches and describes each one (384 numbers each)
          │
          ▼
🔍 ATTENTION MASKING keeps only the 300 most important patches (throws away background)
          │
          ▼
📦 GMM SORTING assigns each patch to one of 12 clusters
          │   (patch at row 5, col 12 → Cluster 3 = "cat_ear")
          ▼
📊 COUNTING: What fraction of patches went to each named cluster?
          │   cat_ear=0.161, cat_eye=0.045, cat_face=0.169, light_fur=0.006
          ▼
🚪 ONECLASSSVM asks: "Does this concept score profile look like a cat's profile?"
          │   Score = +0.82 → YES, it is inside the cat boundary
          ▼
✅ PREDICTION: CAT (82% confidence)
          │
          ▼
🗺️ EXPLANATION: Colour-coded map + bar chart showing exactly what was found and where
```

---

## Connecting to Medical Use

Now replace every "cat" with "disease" and every "not cat" with "healthy":

| Cat System | Medical System |
|---|---|
| Cat photos | X-rays / MRI scans with confirmed diagnosis |
| Bird / airplane photos | Healthy patient scans |
| Cat ear patches | Patches showing irregular lesion boundaries |
| Cat eye patches | Patches showing the dark lesion core |
| Light fur patches | Patches showing unusual tissue texture |
| GMM trained on cat patches | GMM trained on disease scan patches |
| Radiologist labels clusters | Radiologist labels clusters: "lesion margin", "healthy tissue" |
| OneClassSVM trained on cat scores | OneClassSVM trained on disease score profiles |
| "CAT" prediction | "Disease detected" |
| "NOT CAT" prediction | "Healthy — no disease pattern found" |
| Colour map showing cat parts | Colour map showing which regions look like disease features |
| Bar chart of cat concepts | Bar chart showing which disease features were found and how strongly |

**A radiologist using this system can:**
1. See the prediction (positive or negative) with a confidence score
2. See exactly which regions of the scan triggered the alert
3. See which disease features were found and how strongly
4. Agree, disagree, or ask for a second opinion — with actual evidence to review

This is the missing piece in current medical AI systems. Our system explains itself.

---

## Why This Is Clever (The "So What?")

| Problem with Normal AI | What Our System Does |
|---|---|
| "Black box" — no explanation | Explains every decision with named concepts |
| Needs thousands of labeled images of both sick AND healthy patients | Only needs examples of sick patients to train |
| Learns any pattern — even wrong ones like "patients with dark skin" | Learns parts of the object/organ — anatomically meaningful |
| If wrong, no way to know why | If wrong, you can see exactly which feature confused it |
| Cannot tell you WHERE in the image the evidence is | Shows a colour-coded map of exactly where each feature was found |
| Treats the whole image as one thing | Understands the image as a collection of named parts |

---

## Summary in Three Sentences

**What we built:** A computer system that looks at images in tiny pieces, finds recurring visual patterns across many images, asks a human expert to name those patterns, and then uses those named patterns to classify new images and explain exactly which patterns it found and where.

**Why it is special:** Unlike normal AI systems, ours does not just say "sick" or "healthy" — it says *which visual features* it found, *how strongly* they appeared, and *where exactly* in the image they are — like a detective showing their evidence.

**What it is useful for:** Medical image analysis, where a doctor needs to understand and verify the AI's reasoning before trusting it to help diagnose patients.
