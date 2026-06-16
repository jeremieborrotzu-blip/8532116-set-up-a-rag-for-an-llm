# pages/1_Feedback_Viewer.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the parent folder to the Python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import the modules from the parent folder
from utils.database import get_all_interactions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Feedback Viewer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactions and Feedback Viewer")
st.caption("Displays the latest interactions logged in the database.")

# Button to refresh the data
if st.button("🔄 Refresh the data"):
    st.cache_data.clear() # Invalidate the get_all_interactions cache if used

# Retrieve the data (using st.cache_data for caching)
@st.cache_data(ttl=60) # Cache the data for 60 seconds
def load_data():
    logging.info("Loading the interactions from the database for the viewer...")
    interactions_list = get_all_interactions(limit=200) # Increase the limit if needed
    if not interactions_list:
        # Return two empty DataFrames if there is no data
        empty_df = pd.DataFrame()
        return empty_df, empty_df

    # Convert the list of dictionaries to a Pandas DataFrame for easy display
    df = pd.DataFrame(interactions_list)

    # Optional: Improve the presentation of the DataFrame
    # Convert the timestamp to a datetime type if not already done
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Sort by most recent timestamp first
    df = df.sort_values(by='timestamp', ascending=False)
    # Select and rename the columns for clarity
    df_display = df[[
        'timestamp',
        'query',
        'response',
        'feedback',
        'feedback_comment',
        'id', # Keep the ID for reference
        'sources', # Keep the sources for inspection if needed
        'metadata' # Information about the mode used
    ]].rename(columns={
        'timestamp': 'Date & Time (UTC)',
        'query': 'User Question',
        'response': 'Assistant Answer',
        'feedback': 'Feedback',
        'feedback_comment': 'Comment',
        'id': 'Interaction ID',
        'metadata': 'Mode'
    })
    return df_display, df # Also return the original df in case access to the sources is needed

# Load and display the data
try:
    df_display, df_original = load_data()

    if df_display.empty:
        st.warning("No interaction logged in the database for the moment.")
    else:
        st.info(f"{len(df_display)} interactions found.")

        # Create one tab for the statistics and one for the raw data
        tab1, tab2 = st.tabs(["Statistics", "Raw data"])

        with tab1:
            st.subheader("📊 Feedback statistics")

            # Use the numeric feedback values if available

            # Add a column for the numeric feedback value if it exists
            if 'feedback_value' in df_original.columns:
                feedback_values = df_original['feedback_value'].dropna()
            else:
                # Convert the texts to numeric values if the column does not exist
                feedback_values = df_original['feedback'].apply(
                    lambda x: 1 if x == "positive" else 0 if x == "negative" else None
                ).dropna()

            # Count the positive and negative feedback
            if len(feedback_values) > 0:
                positive_count = sum(feedback_values == 1)
                negative_count = sum(feedback_values == 0)
                total_count = len(feedback_values)
                positive_percent = (positive_count / total_count * 100) if total_count > 0 else 0
                negative_percent = (negative_count / total_count * 100) if total_count > 0 else 0

                # Display the statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total feedback", total_count)
                with col2:
                    st.metric("Positive feedback", positive_count, f"{positive_percent:.1f}%")
                with col3:
                    st.metric("Negative feedback", negative_count, f"{negative_percent:.1f}%")

                # Create a bar chart
                feedback_data = pd.DataFrame({
                    'Type': ['Positive', 'Negative'],
                    'Count': [positive_count, negative_count]
                })

                fig = px.bar(
                    feedback_data,
                    x='Type',
                    y='Count',
                    color='Type',
                    color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B'},
                    title="Feedback distribution"
                )

                # Add the percentages on the bars
                fig.update_traces(texttemplate='%{y} (%{y/sum:.1%})', textposition='outside')

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add a chart of feedback evolution over time if there is enough data
                if len(df_original) >= 5:
                    st.subheader("📈 Feedback evolution over time")

                    # Convert the timestamp to datetime if not already done
                    df_original['timestamp'] = pd.to_datetime(df_original['timestamp'])

                    # Create a column for the date (without the time)
                    df_original['date'] = df_original['timestamp'].dt.date

                    # Group by date and count the positive and negative feedback
                    if 'feedback_value' in df_original.columns:
                        # Use the feedback_value column if available
                        feedback_by_date = df_original.groupby('date').apply(
                            lambda x: pd.Series({
                                'positive': sum(x['feedback_value'] == 1),
                                'negative': sum(x['feedback_value'] == 0),
                                'total': len(x)
                            })
                        ).reset_index()
                    else:
                        # Otherwise, use the feedback column
                        feedback_by_date = df_original.groupby('date').apply(
                            lambda x: pd.Series({
                                'positive': sum(x['feedback'] == "positive"),
                                'negative': sum(x['feedback'] == "negative"),
                                'total': len(x)
                            })
                        ).reset_index()

                    # Create an evolution chart
                    fig2 = go.Figure()

                    # Add the lines for the positive and negative feedback
                    fig2.add_trace(go.Scatter(
                        x=feedback_by_date['date'],
                        y=feedback_by_date['positive'],
                        mode='lines+markers',
                        name='Positive',
                        line=dict(color='#00CC96', width=2),
                        marker=dict(size=8)
                    ))

                    fig2.add_trace(go.Scatter(
                        x=feedback_by_date['date'],
                        y=feedback_by_date['negative'],
                        mode='lines+markers',
                        name='Negative',
                        line=dict(color='#EF553B', width=2),
                        marker=dict(size=8)
                    ))

                    # Configure the chart
                    fig2.update_layout(
                        title="Feedback evolution per day",
                        xaxis_title="Date",
                        yaxis_title="Number of feedback",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    # Display the chart
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No feedback has been given yet.")

        with tab2:
            st.subheader("📃 Raw data")
            st.dataframe(
            df_display,
            use_container_width=True,
            # Column configuration to adjust the width and formatting
            column_config={
                "Date & Time (UTC)": st.column_config.DatetimeColumn(
                    format="YYYY-MM-DD HH:mm:ss",
                    width="small"
                ),
                "User Question": st.column_config.TextColumn(width="medium"),
                "Assistant Answer": st.column_config.TextColumn(width="large"),
                "Feedback": st.column_config.TextColumn(width="small"),
                "Comment": st.column_config.TextColumn(width="medium"),
                "Interaction ID": st.column_config.NumberColumn(width="small"),
                "sources": st.column_config.ListColumn(width="medium"), # Display as a list
                "Mode": st.column_config.JsonColumn(width="medium") # Display the metadata as JSON
            },
            hide_index=True # Hide the DataFrame index
        )

        # Optional: Allow viewing the details of an interaction (including the sources)
        st.subheader("🔍 Inspect a specific interaction")
        selected_id = st.selectbox("Select the interaction ID:", options=df_original['id'].tolist())

        if selected_id:
            selected_interaction = df_original[df_original['id'] == selected_id].iloc[0]
            st.write(f"**Question:** {selected_interaction['query']}")
            st.write(f"**Answer:** {selected_interaction['response']}")
            st.write(f"**Feedback:** {selected_interaction['feedback']} {selected_interaction['feedback_comment'] or ''}")

            # Display the metadata (mode, confidence, etc.)
            metadata = selected_interaction['metadata']
            if metadata and isinstance(metadata, dict):
                mode = metadata.get('mode', 'N/A')
                confidence = metadata.get('confidence', 0.0)
                reason = metadata.get('reason', 'N/A')
                st.write(f"**Mode:** {mode} (confidence: {confidence:.2f})")
                st.write(f"**Reason:** {reason}")
            elif metadata:
                st.write("**Metadata:**")
                st.json(metadata)
            st.write("**Sources used during generation:**")
            sources = selected_interaction['sources']
            if sources and isinstance(sources, list):
                 for i, src in enumerate(sources):
                     meta = src.get("metadata", {})
                     with st.expander(f"Source {i+1}: `{meta.get('source', 'N/A')}` (Score: {src.get('score', 0.0):.4f})"):
                         st.text(src.get('text', 'N/A'))
            elif sources:
                 st.json(sources) # Display the raw JSON if it is not a list
            else:
                 st.write("No source logged for this interaction.")


except Exception as e:
    logging.error(f"Error while loading or displaying the data: {e}", exc_info=True)
    st.error(f"An error occurred while displaying the feedback: {e}")
