"""
Phase 8: Gradio demo for the multimodal CVD (stroke) risk model.

Inputs: one MRI brain slice (image upload) + a short clinical form (the
same fields src/data/ehr_features.py uses). Output: a blended risk score
(see src/fusion/inference.py for the real late-fusion inference this
calls -- CNN imaging probability + EHR probability + a tuned alpha) and
the Grad-CAM overlay for the uploaded slice.

Run locally: `python app/gradio_app.py` from the repo root, then open
the printed local URL. See README.md for Hugging Face Spaces deployment
steps once this has been checked out locally.
"""
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "fusion"))
from inference import predict_risk  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

DISCLAIMER = (
    "**This is a portfolio/research demo, not a clinical tool.** Trained entirely on MITRE's "
    "synthetic Coherent Dataset (no real patient data, no PHI) with limited compute -- the imaging "
    "branch saw only 5 training epochs. Cross-validated performance is modest (AUROC ~0.67-0.68), "
    "and the CNN's own Grad-CAM attention often falls outside brain tissue (see notebooks/03_explainability.ipynb). "
    "Do not use this for any real medical decision."
)


def run_app(
    image,
    age, sex, systolic_bp, diastolic_bp,
    total_cholesterol, hdl_cholesterol, bmi, glucose, smoking_status,
):
    if image is None:
        raise gr.Error("Upload an MRI slice image first.")

    clinical_fields = {
        "age": age, "sex": sex, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "total_cholesterol": total_cholesterol, "hdl_cholesterol": hdl_cholesterol,
        "bmi": bmi, "glucose": glucose, "smoking_status": smoking_status,
    }
    result = predict_risk(image, clinical_fields)

    risk_pct = result["risk_score"] * 100
    summary = (
        f"## Predicted stroke risk: {risk_pct:.1f}%\n\n"
        f"| Component | Probability |\n|---|---|\n"
        f"| Imaging model (CNN) | {result['imaging_probability'] * 100:.1f}% |\n"
        f"| EHR model (logistic regression) | {result['ehr_probability'] * 100:.1f}% |\n"
        f"| Blend weight (alpha, imaging share) | {result['alpha']:.2f} |\n\n"
        f"*Risk score = alpha x imaging + (1 - alpha) x EHR.*"
    )
    return summary, result["grad_cam_overlay"]


with gr.Blocks(title="Multimodal Stroke Risk Demo") as demo:
    gr.Markdown("# Multimodal CVD (Stroke) Risk Demo")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="MRI brain slice", type="numpy")
            with gr.Row():
                age_input = gr.Number(label="Age", value=82)
                sex_input = gr.Dropdown(label="Sex", choices=["male", "female"], value="male")
            with gr.Row():
                systolic_input = gr.Number(label="Systolic BP", value=165)
                diastolic_input = gr.Number(label="Diastolic BP", value=85)
            with gr.Row():
                total_chol_input = gr.Number(label="Total cholesterol", value=172.7)
                hdl_chol_input = gr.Number(label="HDL cholesterol", value=48.6)
            with gr.Row():
                bmi_input = gr.Number(label="BMI", value=30.3)
                glucose_input = gr.Number(label="Glucose", value=76.8)
            smoking_input = gr.Dropdown(
                label="Smoking status", choices=["Never smoker", "Former smoker", "Unknown"], value="Never smoker"
            )
            submit_btn = gr.Button("Predict risk", variant="primary")

        with gr.Column():
            risk_output = gr.Markdown(label="Result")
            gradcam_output = gr.Image(label="Grad-CAM overlay (where the CNN is looking)")

    submit_btn.click(
        fn=run_app,
        inputs=[
            image_input, age_input, sex_input, systolic_input, diastolic_input,
            total_chol_input, hdl_chol_input, bmi_input, glucose_input, smoking_input,
        ],
        outputs=[risk_output, gradcam_output],
    )

    gr.Examples(
        examples=[
            [str(EXAMPLES_DIR / "stroke_positive.png"), 82, "male", 165, 85, 172.7, 48.6, 30.3, 76.8, "Never smoker"],
            [str(EXAMPLES_DIR / "stroke_negative.png"), 85, "male", 135, 47, 185.8, 75.8, 27.5, 98.3, "Never smoker"],
        ],
        inputs=[
            image_input, age_input, sex_input, systolic_input, diastolic_input,
            total_chol_input, hdl_chol_input, bmi_input, glucose_input, smoking_input,
        ],
        label="Example patients (real de-identified synthetic cases)",
    )

if __name__ == "__main__":
    demo.launch()
