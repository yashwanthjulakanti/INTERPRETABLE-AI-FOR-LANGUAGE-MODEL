# Interpretable AI for Language Model

This project explores and implements techniques using SHAP, LIME, and Captum to interpret the predictions of BERT on a news classification task. The main goal is to improve the transparency and explainability of AI models.


## Directory Structure
- `Images/`: Directory containing images

## File Description
- `BERT_CAPTUM_SHAP_LIME.ipynb`: Jupyter notebook containing code for model interpretability using LIME, SHAP and CAPTUM.
- Requirements.txt file

## Project Overview
The increasing adoption of language models, such as BERT, has led to a growing need for interpretability in AI systems. Understanding how these models make predictions and providing explanations for their decisions is crucial for building trust and ensuring accountability. This project is an attempt to address this need by exploring and implementing model interpretability techniques using SHAP, LIME, and Captum.

## Model Interpretation Methods Used
- **SHAP (SHapley Additive exPlanations)**: A game theoretic approach to explain the output of any machine learning model.
- **LIME (Local Interpretable Model-Agnostic Explanations)**: Explains the predictions of any machine learning classifier in an interpretable and faithful manner by learning an interpretable model locally around the prediction.
- **Captum**: Provides model interpretability techniques for PyTorch models, allowing us to understand the importance of model inputs in terms of their contributions to the final prediction.

## How to Run
1. Clone the repository and navigate to the downloaded folder.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the notebooks: `BERT_CAPTUM_SHAP_LIME.ipynb`

## References
1. Lundberg, S.M., Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). "Why should I trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).
3. Kapishnikov, A., Engelhardt, M., Soto, M., et al. (2020). Captum: A Unified and Generic API for Model Interpretability. arXiv preprint arXiv:2009.07813.

## Acknowledgments
This project was inspired by the need for model interpretability in AI systems.
