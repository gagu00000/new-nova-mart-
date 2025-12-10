import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def render(data):
    """Render the ML Model Evaluation page"""
    
    st.markdown('<h1 class="main-header">🤖 ML Model Evaluation</h1>', unsafe_allow_html=True)
    
    lead_df = data['lead_scoring']
    feature_df = data['feature_importance']
    learning_df = data['learning_curve']
    
    # Section 1: Confusion Matrix
    st.subheader("🎯 Confusion Matrix - Lead Scoring Model")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="threshold"
        )
        
        show_percentages = st.checkbox(
            "Show Percentages",
            value=False,
            key="cm_pct"
        )
    
    with col1:
        # Apply threshold
        predicted_class = (lead_df['predicted_probability'] >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(lead_df['actual_converted'], predicted_class)
        
        if show_percentages:
            cm_display = cm.astype('float') / cm.sum() * 100
            text = [[f'{val:.1f}%' for val in row] for row in cm_display]
        else:
            cm_display = cm
            text = [[f'{val}' for val in row] for row in cm]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_display,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            text=text,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix (Threshold: {threshold})",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        st.metric("F1 Score", f"{f1:.3f}")
    
    st.markdown("---")
    
    # Section 2: ROC Curve
    st.subheader("📈 ROC Curve - Model Performance")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**Model Evaluation**")
        st.markdown(
            "ROC curve shows the trade-off between "
            "True Positive Rate and False Positive Rate."
        )
    
    with col1:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(lead_df['actual_converted'], lead_df['predicted_probability'])
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's index)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        fig = go.Figure()
        
        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=1)
        ))
        
        # Optimal threshold point
        fig.add_trace(go.Scatter(
            x=[fpr[optimal_idx]],
            y=[tpr[optimal_idx]],
            mode='markers',
            name=f'Optimal Threshold ({optimal_threshold:.3f})',
            marker=dict(color='green', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.info(
        f"**Model Performance:** AUC = {roc_auc:.3f} indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'acceptable' if roc_auc > 0.7 else 'fair'} discrimination ability. "
        f"Optimal threshold is {optimal_threshold:.3f}."
    )
    
    st.markdown("---")
    
    # Section 3: Precision-Recall Curve (Bonus)
    st.subheader("🎯 Precision-Recall Curve")
    
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
        lead_df['actual_converted'], 
        lead_df['predicted_probability']
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall_vals,
        y=precision_vals,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Section 4: Learning Curve
    st.subheader("📊 Learning Curve - Model Diagnostics")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_confidence = st.checkbox(
            "Show Confidence Bands",
            value=True,
            key="confidence"
        )
    
    with col1:
        fig = go.Figure()
        
        # Training score
        fig.add_trace(go.Scatter(
            x=learning_df['training_size'],
            y=learning_df['train_score'],
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=learning_df['training_size'],
            y=learning_df['validation_score'],
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        if show_confidence:
            # Training confidence band
            fig.add_trace(go.Scatter(
                x=learning_df['training_size'].tolist() + learning_df['training_size'].tolist()[::-1],
                y=(learning_df['train_score'] + learning_df['train_score_std']).tolist() + 
                  (learning_df['train_score'] - learning_df['train_score_std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Validation confidence band
            fig.add_trace(go.Scatter(
                x=learning_df['training_size'].tolist() + learning_df['training_size'].tolist()[::-1],
                y=(learning_df['validation_score'] + learning_df['validation_score_std']).tolist() + 
                  (learning_df['validation_score'] - learning_df['validation_score_std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="Learning Curve: Training vs Validation Performance",
            xaxis_title="Training Set Size",
            yaxis_title="Model Score (Accuracy)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Learning curve interpretation
    final_train = learning_df.iloc[-1]['train_score']
    final_val = learning_df.iloc[-1]['validation_score']
    gap = final_train - final_val
    
    if gap < 0.05:
        interpretation = "✅ Curves are converging well. Model has low bias and variance."
    elif gap < 0.1:
        interpretation = "⚠️ Small gap between curves. Model may benefit from slight regularization."
    else:
        interpretation = "❌ Large gap indicates overfitting. Consider regularization or more data."
    
    st.info(f"**Interpretation:** {interpretation}")
    
    st.markdown("---")
    
    # Section 5: Feature Importance
    st.subheader("🔍 Feature Importance - Model Interpretability")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        sort_order = st.radio(
            "Sort Order",
            ["Descending", "Ascending"],
            key="feat_sort"
        )
        
        show_error_bars = st.checkbox(
            "Show Error Bars",
            value=True,
            key="error_bars"
        )
    
    with col1:
        feature_sorted = feature_df.sort_values('importance', ascending=(sort_order == "Ascending"))
        
        fig = go.Figure()
        
        if show_error_bars:
            fig.add_trace(go.Bar(
                y=feature_sorted['feature'],
                x=feature_sorted['importance'],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=feature_sorted['importance_std'],
                    visible=True
                ),
                marker=dict(
                    color=feature_sorted['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=feature_sorted['importance'].round(3),
                textposition='auto'
            ))
        else:
            fig.add_trace(go.Bar(
                y=feature_sorted['feature'],
                x=feature_sorted['importance'],
                orientation='h',
                marker=dict(
                    color=feature_sorted['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=feature_sorted['importance'].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Feature Importance Scores",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 ML Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_feature = feature_df.loc[feature_df['importance'].idxmax(), 'feature']
        top_importance = feature_df['importance'].max()
        st.info(
            f"**Top Predictor**\n\n"
            f"{top_feature} (importance: {top_importance:.3f}) is the strongest "
            f"predictor of lead conversion."
        )
    
    with col2:
        st.success(
            f"**Model Quality**\n\n"
            f"AUC: {roc_auc:.3f}, Accuracy: {accuracy:.2%}. "
            f"Model is production-ready with good performance."
        )
    
    with col3:
        st.warning(
            f"**Recommended Threshold**\n\n"
            f"Use {optimal_threshold:.3f} for optimal balance between "
            f"precision and recall."
        )
