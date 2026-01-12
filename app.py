import streamlit as st
import pandas as pd
import os
from datetime import datetime
from main import ProcurementTaxonomyPipeline
from hierarchical_classifier import HierachicalTaxonomyClassifier
import gc
import io

# Page configuration
st.set_page_config(
    page_title="Taxonomy Classifier",
    page_icon="🏷️",
    layout="wide"
)

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# Title
st.title("🏷️ Procurement Taxonomy Classifier")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["🎓 Train Model", "🔮 Make Predictions"])

# ============================================================================
# TAB 1: TRAIN MODEL
# ============================================================================
with tab1:
    st.header("Train New Model")
    st.markdown("Upload your training data and configure model parameters.")

    # File upload
    st.subheader("1. Upload Training Data")
    training_file = st.file_uploader(
        "Upload Excel file with labeled data",
        type=['xlsx', 'xls'],
        key='train_file',
        help="Excel file must contain 'Overall' sheet with labeled taxonomy data"
    )

    # Parameters
    st.subheader("2. Configure Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test split size",
            min_value=0.1,
            max_value=0.3,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing (default: 20%)"
        )

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Minimum confidence for auto-accepting predictions (default: 0.5)"
        )

    with col2:
        # Advanced parameters (expandable)
        with st.expander("⚙️ Advanced Parameters"):
            alpha_l1 = st.number_input("Alpha L1", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
            alpha_l2 = st.number_input("Alpha L2", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
            alpha_l3 = st.number_input("Alpha L3", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
            max_features = st.number_input("Max features", value=5000, min_value=1000, max_value=10000, step=500)

    # Train button
    st.subheader("3. Train Model")

    if st.button("🚀 Train Model", type="primary", disabled=(training_file is None)):
        if training_file is not None:
            # Save uploaded file temporarily
            temp_input_path = f"temp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            with open(temp_input_path, 'wb') as f:
                f.write(training_file.read())

            try:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Initialize pipeline
                status_text.text("Initializing pipeline...")
                progress_bar.progress(10)

                pipeline = ProcurementTaxonomyPipeline(confidence_threshold=confidence_threshold)

                # Override classifier parameters if advanced settings changed
                if alpha_l1 != 0.1 or alpha_l2 != 0.1 or alpha_l3 != 0.1 or max_features != 5000:
                    from preprocessing import TextPreprocessor
                    pipeline.classifier = HierachicalTaxonomyClassifier(
                        alpha_l1=alpha_l1,
                        alpha_l2=alpha_l2,
                        alpha_l3=alpha_l3,
                        max_features=max_features,
                        ngram_range=(1,2),
                        min_df=2,
                        text_preprocessor=TextPreprocessor()
                    )

                # Load and prepare data
                status_text.text("Loading and preprocessing data...")
                progress_bar.progress(20)

                df = pd.read_excel(temp_input_path, sheet_name='Overall')
                labeled_df, unlabeled_df = pipeline.identify_new_rows(df)

                # Train model
                status_text.text("Training model... This may take a few minutes.")
                progress_bar.progress(40)

                evaluation_results = pipeline.train_model(labeled_df, test_size=test_size)

                # Save model
                status_text.text("Saving model...")
                progress_bar.progress(90)

                model_save_path = f"taxonomy_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                pipeline.classifier.save(model_save_path)

                gc.collect()

                # Store in session state
                st.session_state.trained_model = pipeline.classifier
                st.session_state.model_path = model_save_path

                progress_bar.progress(100)
                status_text.text("Training complete!")

                # Display results
                st.success(f"✅ Model trained successfully and saved to: `{model_save_path}`")

                # Show evaluation metrics
                st.subheader("Model Performance")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("L1 Accuracy", f"{evaluation_results['l1']['accuracy']:.1%}")
                    st.metric("L1 F1-Score", f"{evaluation_results['l1']['f1_weighted']:.3f}")

                with col2:
                    st.metric("L2 Accuracy", f"{evaluation_results['l2']['accuracy']:.1%}")
                    st.metric("L2 F1-Score", f"{evaluation_results['l2']['f1_weighted']:.3f}")

                with col3:
                    st.metric("L3 Accuracy", f"{evaluation_results['l3']['accuracy']:.1%}")
                    st.metric("L3 F1-Score", f"{evaluation_results['l3']['f1_weighted']:.3f}")

                # Training data info
                st.info(f"📊 Training data: {len(labeled_df)} labeled samples, {len(unlabeled_df)} unlabeled samples")

                # Download model
                with open(model_save_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name=model_save_path,
                        mime="application/octet-stream"
                    )

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

            finally:
                # Clean up temp file
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)

# ============================================================================
# TAB 2: MAKE PREDICTIONS
# ============================================================================
with tab2:
    st.header("Make Predictions")
    st.markdown("Upload data and a trained model to generate taxonomy predictions.")

    # File uploads
    st.subheader("1. Upload Files")

    col1, col2 = st.columns(2)

    with col1:
        prediction_file = st.file_uploader(
            "Upload Excel file with data to classify",
            type=['xlsx', 'xls'],
            key='pred_file',
            help="Excel file with 'Overall' sheet containing unlabeled data"
        )

    with col2:
        model_file = st.file_uploader(
            "Upload trained model (.joblib)",
            type=['joblib'],
            key='model_file',
            help="Previously trained model file"
        )

        # Option to use model from training tab
        if st.session_state.trained_model is not None:
            use_session_model = st.checkbox(
                f"Use model from training tab: `{st.session_state.model_path}`",
                value=True
            )
        else:
            use_session_model = False

    # Parameters
    st.subheader("2. Prediction Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pred_confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.05,
            key='pred_threshold',
            help="Minimum confidence for auto-accepting predictions"
        )

        enable_highlighting = st.checkbox(
            "Enable granular highlighting",
            value=True,
            help="Highlight low-confidence predictions in the output Excel file"
        )

    with col2:
        output_filename = st.text_input(
            "Output filename (optional)",
            value="",
            placeholder=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            help="Leave empty for auto-generated filename"
        )

    # Predict button
    st.subheader("3. Generate Predictions")

    can_predict = prediction_file is not None and (model_file is not None or use_session_model)

    if st.button("🔮 Generate Predictions", type="primary", disabled=not can_predict):
        # Save uploaded files temporarily
        temp_input_path = f"temp_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        temp_model_path = f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"

        with open(temp_input_path, 'wb') as f:
            f.write(prediction_file.read())

        # Handle model
        if use_session_model:
            classifier = st.session_state.trained_model
        else:
            with open(temp_model_path, 'wb') as f:
                f.write(model_file.read())
            classifier = HierachicalTaxonomyClassifier.load(temp_model_path)

        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize pipeline
            status_text.text("Loading data...")
            progress_bar.progress(10)

            pipeline = ProcurementTaxonomyPipeline(confidence_threshold=pred_confidence_threshold)
            pipeline.classifier = classifier

            # Load data
            df = pd.read_excel(temp_input_path, sheet_name='Overall')
            labeled_df, unlabeled_df = pipeline.identify_new_rows(df)

            progress_bar.progress(30)

            # Make predictions
            if len(unlabeled_df) > 0:
                status_text.text(f"Generating predictions for {len(unlabeled_df)} rows...")
                progress_bar.progress(50)

                predicted_df = pipeline.predict_new_rows(unlabeled_df)

                progress_bar.progress(70)
                status_text.text("Merging results...")

                # Merge results
                result_df = pipeline.merge_results(df, labeled_df, predicted_df)

                progress_bar.progress(85)
                status_text.text("Saving results...")

                # Save results
                if not output_filename:
                    output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                else:
                    output_path = output_filename if output_filename.endswith('.xlsx') else f"{output_filename}.xlsx"

                pipeline.save_results_to_excel(result_df, output_path, highlight_low_confidence=enable_highlighting)

                progress_bar.progress(100)
                status_text.text("Predictions complete!")

                # Display summary
                st.success(f"✅ Predictions generated successfully!")

                st.subheader("Prediction Summary")

                # Calculate statistics
                auto_accept_count = predicted_df['Auto_Accept'].sum()
                manual_review_count = len(predicted_df) - auto_accept_count
                avg_confidence = predicted_df['Combined_Confidence'].mean()
                min_confidence = predicted_df['Combined_Confidence'].min()
                max_confidence = predicted_df['Combined_Confidence'].max()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Predictions", len(predicted_df))

                with col2:
                    st.metric(
                        "Auto-Accept",
                        auto_accept_count,
                        delta=f"{auto_accept_count/len(predicted_df)*100:.1f}%"
                    )

                with col3:
                    st.metric(
                        "Manual Review",
                        manual_review_count,
                        delta=f"{manual_review_count/len(predicted_df)*100:.1f}%",
                        delta_color="inverse"
                    )

                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")

                st.info(f"📊 Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")

                # Preview predictions
                st.subheader("Preview (First 10 Predictions)")

                preview_cols = [
                    'L1_Prediction', 'L2_Prediction', 'L3_Prediction',
                    'L1_Confidence', 'L2_Confidence', 'L3_Confidence',
                    'Combined_Confidence', 'Auto_Accept', 'Review_Reason'
                ]

                preview_df = predicted_df[preview_cols].head(10)

                # Style the dataframe
                def highlight_confidence(val):
                    if pd.isna(val):
                        return ''
                    try:
                        val = float(val)
                        if val >= 0.7:
                            return 'background-color: #d4edda'  # Green
                        elif val >= pred_confidence_threshold:
                            return 'background-color: #fff3cd'  # Yellow
                        else:
                            return 'background-color: #f8d7da'  # Red
                    except:
                        return ''

                styled_df = preview_df.style.applymap(
                    highlight_confidence,
                    subset=['L1_Confidence', 'L2_Confidence', 'L3_Confidence', 'Combined_Confidence']
                )

                st.dataframe(styled_df, use_container_width=True)

                # Download results
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Results",
                        data=f,
                        file_name=output_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            else:
                st.warning("⚠️ No unlabeled rows found in the input file. All rows already have taxonomy labels.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

        finally:
            # Clean up temp files
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if not use_session_model and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Procurement Taxonomy Classifier**

    A hierarchical machine learning system for automatically classifying procurement contracts into a 3-level taxonomy (L1 → L2 → L3).

    **Features:**
    - Hierarchical classification
    - Confidence-based predictions
    - Granular highlighting
    - Official taxonomy integration
    """)

    st.markdown("---")

    if st.session_state.trained_model is not None:
        st.success("✅ Model loaded in session")
        st.caption(f"Path: `{st.session_state.model_path}`")
    else:
        st.info("No model loaded in current session")

    st.markdown("---")
    st.caption("Built with Streamlit")

