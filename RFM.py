import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Function to map cluster numbers to meaningful labels
def assign_cluster_labels(cluster_numbers):
    label_mapping = {
        0: 'Hibernating',
        1: 'At Risk',
        2: 'Champions',
        3: 'Loyal',
        4: 'Best',
        5: 'Greedy'
    }
    return [label_mapping.get(cluster, 'Unknown') for cluster in cluster_numbers]

# Title of the app
st.title('Customer Segmentation Using RFM Analysis')

# Sidebar for file upload
st.sidebar.header('Upload CSV File')
st.sidebar.markdown("Upload a CSV file containing customer transaction data. Ensure it has columns for Recency, Frequency, and Monetary values.")

# File uploader widget
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty. Please upload a valid CSV file.")
        else:
            st.write("Dataset Preview:")
            st.dataframe(df.head())

            # Select columns for RFM analysis
            st.sidebar.header('Select Columns for RFM Analysis')
            recency_col = st.sidebar.selectbox('Select Recency Column', df.columns)
            frequency_col = st.sidebar.selectbox('Select Frequency Column', df.columns)
            monetary_col = st.sidebar.selectbox('Select Monetary Column', df.columns)

            # Check if selected columns exist in the dataset
            if recency_col not in df.columns or frequency_col not in df.columns or monetary_col not in df.columns:
                st.error("Selected columns are not in the dataset. Please check your selection.")
            else:
                # Allow the user to select the number of clusters
                num_clusters = st.sidebar.slider("Select the number of clusters", min_value=2, max_value=10, value=4)

                # Scaling the RFM values
                scaler = StandardScaler()
                df_rfm = df[[recency_col, frequency_col, monetary_col]]
                df_rfm_scaled = scaler.fit_transform(df_rfm)

                # Allow user to select the clustering algorithm
                st.sidebar.header('Select Clustering Algorithm')
                cluster_algo = st.sidebar.selectbox('Select Algorithm', ('KMeans', 'Hierarchical Clustering'))

                # Dynamic file name based on selected algorithm
                if cluster_algo == 'KMeans':
                    # Apply KMeans clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    df['Cluster'] = kmeans.fit_predict(df_rfm_scaled)  # KMeans Clusters
                    df['Cluster_Label'] = assign_cluster_labels(df['Cluster'])  # Assign meaningful labels
                    file_name = "segmented_data_KMeans.csv"  # Set file name based on algorithm
                    # Define columns for the CSV file in the KMeans case
                    columns_to_include = [
                        'CustomerID', 'MerchantName', 'Category', 'TransactionRank', 'TransactionRedeemedPoints', 
                        'TransactionValue', 'TransactionFrom(days)', 'CustomerLastTransactionFrom(days)',
                        'Recency', 'Frequency', 'Monetary', 'Cluster', 'Cluster_Label'
                    ]
                elif cluster_algo == 'Hierarchical Clustering':
                    # Apply Hierarchical Clustering
                    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
                    df['Cluster'] = hierarchical.fit_predict(df_rfm_scaled)  # Hierarchical Clusters
                    df['Cluster_Label'] = assign_cluster_labels(df['Cluster'])  # Assign meaningful labels
                    file_name = "segmented_data_Hierarchical.csv"  # Set file name based on algorithm
                    # Define columns for the CSV file in the Hierarchical Clustering case
                    columns_to_include = [
                        'CustomerID', 'MerchantName', 'Category', 'TransactionRank', 'TransactionRedeemedPoints', 
                        'TransactionValue', 'TransactionFrom(days)', 'CustomerLastTransactionFrom(days)',
                        'Recency', 'Frequency', 'Monetary', 'Cluster', 'Cluster_Label'
                    ]

                # Generate the CSV file for download
                segmented_df = df[columns_to_include]
                
                # Encode to CSV
                segmented_csv = segmented_df.to_csv(index=False).encode('utf-8')

                # Display the segmented data
                st.write("Segmented Data Preview:")
                st.dataframe(segmented_df)  # Now showing the selected columns only

                # Visualization of clusters
                st.header('Cluster Visualization')
                fig, ax = plt.subplots()
                scatter = ax.scatter(df[recency_col], df[frequency_col], c=df['Cluster'], cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                plt.xlabel('Recency')
                plt.ylabel('Frequency')
                st.pyplot(fig)

                # Additional visualizations using seaborn (Cluster Distributions)
                st.header('Cluster Value Distributions')
                for cluster in df['Cluster'].unique():
                    st.write(f"Cluster {cluster} Distributions:")
                    # Create a figure and axes
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df[df['Cluster'] == cluster][[recency_col, frequency_col, monetary_col]], ax=ax)
                    # Pass the figure to st.pyplot
                    st.pyplot(fig)

                # Allow user to download segmented data with dynamic file name
                st.sidebar.header('Download Segmented Data')
                st.sidebar.download_button(label="Download CSV", data=segmented_csv, file_name=file_name, mime='text/csv')

    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
