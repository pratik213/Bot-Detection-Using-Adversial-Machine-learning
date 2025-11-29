# Improving Bot Detection: Theory & Strategies

You noticed a high number of **False Negatives (FN)** in your confusion matrix. This means:
> **The IP is a known bot (`ua_bot_flag=1`), but the model classified it as Normal (`is_anomaly=0`).**

Here is the theory on why this happens and how to fix it.

## 1. Why do False Negatives happen?

### A. The "Stealthy Bot" Problem
Isolation Forest works by finding **outliers**—data points that are "few and different".
If a bot is well-behaved (e.g., it makes 1 request every 10 minutes, visits normal pages), it looks statistically identical to a human.
*   **Result**: The model sees it as part of the "normal" cluster.
*   **Theory**: You cannot detect what you cannot distinguish. If the features (request rate, path entropy) are the same as a human's, unsupervised anomaly detection will fail.

### B. Contamination Parameter
The `contamination` parameter tells the model "Roughly X% of this data is bad".
*   If you set `contamination=0.05` (5%), but 15% of your traffic is bots, the model is **forced** to label the "least obvious" 10% of bots as normal.
*   **Fix**: Increasing contamination reduces False Negatives but increases False Positives (flagging real humans as bots).

### C. Unsupervised vs. Supervised
Isolation Forest is **Unsupervised**. It doesn't know what a "bot" is; it only knows what "rare" is.
*   Many "known bots" (like Googlebot) behave very consistently. They are not "anomalous" in terms of behavior; they are just *automated*.
*   If your goal is to catch things that *look* like your known bots, Unsupervised Learning is the wrong tool.

## 2. Strategies to Improve

### Strategy A: Switch to Supervised Learning (Recommended)
Since you have `ua_bot_flag`, you have **labels**. You can train a classifier (Random Forest, XGBoost) to explicitly learn: *"What does a bot look like?"*
*   **Pros**: Will drastically reduce False Negatives on bots that behave like the ones you've seen.
*   **Cons**: Might miss completely new types of bots (zero-day attacks).

### Strategy B: Feature Engineering (The "Fingerprint" Approach)
If you stick with Anomaly Detection, you need features that force the bots to stand out.
1.  **Periodicity**: Humans are random; bots are precise. Calculate the *variance* of time intervals. Low variance = Bot.
2.  **Session Depth**: Bots often hit 1 page and leave, or 10,000 pages. Humans usually hit 5-20.
3.  **Resource Loading**: Real browsers load CSS/JS/Images. Bots often only hit the HTML. (Requires logs to show resource types).

### Strategy C: Hybrid Approach
1.  **Rule-Based**: If `User-Agent` contains "bot", flag it immediately. Don't use ML for this.
2.  **Anomaly Detection**: Use ML *only* for the IPs that pass the User-Agent check. This focuses the model on finding **hidden** bots (the ones pretending to be human).

## 3. Next Steps

I recommend **Strategy A (Supervised Learning)** if you want to maximize accuracy against the bots you currently know about.

We can create a `supervised_bot_detection.py` script that:
1.  Uses `ua_bot_flag` as the **target variable** (y).
2.  Trains a **Random Forest Classifier**.
3.  Evaluates with Cross-Validation.

This usually yields much better F1-scores than Isolation Forest for this specific dataset.
