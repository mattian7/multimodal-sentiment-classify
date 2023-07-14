# Self-SC: Learning Modality-Specific Classifications with Self-Supervised Clustering for Multimodal Sentiment Analysis

This is the official repository for the  fifth experiment of Contemporary Artificial Intelligence Class.

## Model

![](https://raw.githubusercontent.com/mattian7/figure/main/img/ai5.png)

## Performance

|      | feature-based version | decision-based version |    ours    |
| :--: | :-------------------: | :--------------------: | :--------: |
| ACC  |        69.75%         |         70.75%         | **73.25%** |

## Repository Structure

```
.
├── README.md
├── data
├── run.py  # Entrance of inference and training
├── model.py
├── util.py
├── test_without_label.txt
├── train.txt
├── result.txt
├── checkpoints
│   └── model.pth
```

## Usage

1. Clone this repo and install requirements.

   ```bash
   git clone https://github.com/mattian7/multimodal-sentiment-classify.git
   cd multimodal-sentiment-classify
   conda create --name self_sc python=3.8
   source activate self_sc
   pip install -r requirements.txt
   ```

2. Train model

   ```bash
   python run.py
   ```

   The optional parameters and its meanings are as follows:

   - --mode：Which modality of data do you want to use. You can choose one from: text, image, all. For instance, if you only want to use text data, then use `--mode text`
   - --weight_decay：A coefficient helps control the complexity of the model and reduces the risk of overfitting.
   - --epochs：How many epochs do you want to train the model.
   - --test：Whether only run the model on the test dataset or not. Choose 1 for only test. Choose 0 for only train.
   - --lr：Value of learning rate.
   - --fusion_dropout：P value of dropout layer in fusion part.
   - --text_dropout：Pvalue of dropout layer in text part.
   - --image_dropout：P value of dropout layer in image part.
   - --text_dim：The size of full connection layer of text part.
   - --image_dim：The size of full connection layer of image part.

3. Inference on test dataset

   ```bash
   python run.py --test 1
   ```

4. Ablation study (only text)

   ```
   python run.py --mode text
   ```

5. Ablation study (only image)

   ```
   python run.py --mode image
   ```

## References

The following papers and repositories help me to complete this project.

**Kaur, Ramandeep, and Sandeep Kautish. "Multimodal sentiment analysis: A survey and comparison." *Research Anthology on Implementing Sentiment Analysis Across Multiple Disciplines* (2022): 1846-1870.**

**Yu, Wenmeng, et al. "Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 35. No. 12. 2021.**

Self-MM：[thuiar/Self-MM: Codes for paper "Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis" (github.com)](https://github.com/thuiar/Self-MM/tree/main)

