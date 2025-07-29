import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Set page configuration for better layout
st.set_page_config(layout="wide", page_title="S&P 500 Companies: Sector, Location, and Diversity Analysis")


# ------------- S&P 500 COMPANIES: SECTOR, LOCATION, AND DIVERSITY ANALYSIS ----------

st.title("S&P 500 Companies: Sector, Location, and Diversity Analysis")
st.markdown("This application analyzes the S&P 500 companies based on their GICS sector, headquarters location, and sub-industry diversity.")

# Function to load data with caching
@st.cache_data
def load_sp500_data():
    """Scrapes S&P 500 companies from Wikipedia and performs initial cleaning."""
    try:
        st.info("Attempting to scrape S&P 500 data from Wikipedia...")
        scraper = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = scraper[0]
        st.success("S&P 500 data scraped successfully!")

        # Save to CSV for future direct loading (optional, as caching handles this too)
        df.to_csv("sp500.csv", index=False)
        return df
    except Exception as e:
        st.error(f"Error scraping data from Wikipedia: {e}")
        st.warning("Attempting to load data from local 'sp500.csv' file...")
        try:
            df = pd.read_csv("sp500.csv")
            st.success("DataFrame loaded successfully from sp500.csv")
            return df
        except FileNotFoundError:
            st.error("File 'sp500.csv' not found. Please ensure the file exists or scraping works.")
            st.stop() # Stops the app execution if data cannot be loaded
        except Exception as e_csv:
            st.error(f"An error occurred while loading sp500.csv: {e_csv}")
            st.stop()
    return pd.DataFrame() # Return empty DataFrame if all fails


# Load the data
df = load_sp500_data()

if not df.empty:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    st.write(f"Number of Companies: {len(df)}")
    st.write(f"Number of Columns: {len(df.columns)}")
    # st.write("Dataframe Information:")
    # st.text(df.info()) # df.info() prints to console, harder to capture. Displaying head and columns is often enough.
    st.write("Column Names:")
    st.dataframe(pd.DataFrame(df.columns.tolist(), columns=["Column Name"])) # Display column names nicely



    # ------------- CLEANING UP COLUMN NAMES ---------------

    st.subheader("Cleaning Column Names")
    df.columns = (df.columns
                    .str.strip()
                    .str.replace('\n', '', regex=False)
                    .str.replace('-', '_', regex=False)
                    .str.replace(' ', '_', regex=False)
                    .str.lower())
    st.write("Cleaned Column Names:")
    st.dataframe(pd.DataFrame(df.columns.tolist(), columns=["Cleaned Column Name"]))



    # ------------ CLEANING DATA WITHIN SPECIFIC COLUMNS --------------
    st.subheader("Cleaning Data Within Specific Columns")
    columns_to_clean = [
        'security',
        'symbol',
        'gics_sector',
        'gics_sub_industry',
        'headquarters_location'
    ]

    for col_name in columns_to_clean:
        if col_name in df.columns:
            df[col_name] = (df[col_name]
                            .astype(str)
                            .str.replace(r'\[.*?\]', '', regex=True)
                            .str.strip()
                            .replace('nan', pd.NA))
        else:
            st.warning(f"Warning: Column '{col_name}' not found for data cleaning.")

    st.write("DataFrame after basic data cleaning (first 5 rows): ")
    st.dataframe(df.head())



    # ------------ SPLITTING HEADQUARTERS LOCATION INTO CITY AND STATE ------------

    st.subheader("Splitting Headquarters Location into City and State")
    def split_headquarters_oneliner(df_input): # Use a different name like df_input to avoid confusion
        clean_location = df_input['headquarters_location'].fillna('').astype(str)
        split_cols = clean_location.str.rsplit(',', n=1, expand=True)
        df_input['headquarters_city'] = split_cols[0].str.strip()
        df_input['headquarters_state'] = split_cols[1].str.strip().fillna('') if split_cols.shape[1] > 1 else ''
        df_input['headquarters_state'] = df_input['headquarters_state'].replace('', 'Uknown')
        return df_input

    df = split_headquarters_oneliner(df)
    st.write("DataFrame after splitting headquarters location (relevant columns):")
    st.dataframe(df[['security','symbol', 'gics_sector', 'gics_sub_industry','headquarters_location', 'headquarters_city', 'headquarters_state']].head())

    st.write("Unique states found (top 10):")
    st.dataframe(df['headquarters_state'].value_counts().head(10))

    # --- SETTING THE STYLE OF THE PLOT ---
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")



    # ------------ ANALYZE AND VISUALIZE SECTOR DISTRIBUTION ------------
    st.header("1. GICS Sector Distribution")
    def analyze_sector_distribution(df_input, column="gics_sector", figsize=(12, 8)):
        sector_counts = df_input[column].value_counts()
        sector_percentages = df_input[column].value_counts(normalize=True) * 100

        sector_summary = pd.DataFrame({
            'Count': sector_counts,
            'Percentage': sector_percentages.round(2)
        })

        st.subheader("GICS Sector Distribution Summary:")
        st.dataframe(sector_summary)

        fig, ax = plt.subplots(figsize=figsize)
        bars = sns.barplot(
            y=sector_counts.index,
            x=sector_counts.values,
            ax=ax,
            orient='h'
        )

        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            percentage = sector_percentages.iloc[i]
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} ({percentage:.1f}%)',
                    ha='left', va='center', fontweight='bold')

        ax.set_title('S&P 500 Companies by GICS Sector', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Companies', fontsize=12)
        ax.set_ylabel('GICS Sector', fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        return fig, sector_summary # Return the figure object

    fig_sector, sector_summary_data = analyze_sector_distribution(df.copy()) # Pass a copy to avoid modifying original df
    st.pyplot(fig_sector) # Display the plot in Streamlit




    # ------------ ANALYZE AND VISUALIZE STATE HEADQUARTERS DISTRIBUTION ------------
    st.header("2. State Headquarters Distribution")
    def analyze_state_distribution(df_input, column='headquarters_state', top_n=10, figsize=(12, 8)):
        state_counts = df_input[column].value_counts()
        state_percentages = df_input[column].value_counts(normalize=True) * 100

        top_states = state_counts.head(top_n)
        top_percentages = state_percentages.head(top_n)

        state_summary = pd.DataFrame({
            'Count': top_states,
            'Percentage': top_percentages.round(2),
            'Cumulative %': top_percentages.cumsum().round(2)
        })

        st.subheader(f"Top {top_n} States by S&P 500 Headquarters:")
        st.dataframe(state_summary)
        st.write(f"Top {top_n} states represent {state_summary['Cumulative %'].iloc[-1]:.1f}% of all S&P 500 companies")

        fig, ax = plt.subplots(figsize=figsize)
        bars = sns.barplot(
            x=top_states.values,
            y=top_states.index,
            palette='plasma',
            ax=ax
        )

        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            percentage = top_percentages.iloc[i]
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} ({percentage:.1f}%)',
                    ha='left', va='center', fontweight='bold', fontsize=10)

        ax.set_title(f'Top {top_n} States with S&P 500 Company Headquarters',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Companies', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)

        return fig, state_summary

    fig_state, state_summary_data = analyze_state_distribution(df.copy())
    st.pyplot(fig_state)



    # ------------ CONCENTRATION ANALYSIS ------------
    st.subheader("Geographic Concentration Analysis")
    def show_geographic_concentration(df_input, column='headquarters_state', top_n=10):
        total_companies = len(df_input)
        top_states_count = df_input[column].value_counts().head(top_n).sum()
        others_count = total_companies - top_states_count
        others_percentage = (others_count / total_companies) * 100

        st.write(f"Top {top_n} states: {top_states_count} companies ({(top_states_count/total_companies)*100:.1f}%)")
        st.write(f"Other states: {others_count} companies ({others_percentage:.1f}%)")
        st.write(f"Total unique states: {df_input[column].nunique()}")

    show_geographic_concentration(df.copy())



    # ------------ ANALYZE AND VISUALIZE GICS SUB-INDUSTRY DISTRIBUTION ------------
    st.header("3. GICS Sub-Industry Distribution")
    num_unique_subindustries = df['gics_sub_industry'].nunique()
    st.write(f"Number of unique GICS Sub-Industries: {num_unique_subindustries}")

    def analyze_subindustry_distribution(df_input, column='gics_sub_industry', top_n=10, figsize=(14, 10)):
        total_unique = df_input[column].nunique()
        st.write(f"Total unique GICS Sub-Industries in S&P 500: {total_unique}")

        subindustry_counts = df_input[column].value_counts()
        subindustry_percentages = df_input[column].value_counts(normalize=True) * 100

        top_subindustries = subindustry_counts.head(top_n)
        top_percentages = subindustry_percentages.head(top_n)

        subindustry_summary = pd.DataFrame({
            'Count': top_subindustries,
            'Percentage': top_percentages.round(2),
            'Cumulative %': top_percentages.cumsum().round(2)
        })

        st.subheader(f"Top {top_n} GICS Sub-Industries:")
        st.dataframe(subindustry_summary)

        companies_in_top = top_subindustries.sum()
        companies_in_others = len(df_input) - companies_in_top
        st.subheader("Diversity Analysis:")
        st.write(f"Top {top_n} sub-industries: {companies_in_top} companies ({(companies_in_top/len(df_input))*100:.1f}%)")
        st.write(f"Other {total_unique - top_n} sub-industries: {companies_in_others} companies ({(companies_in_others/len(df_input))*100:.1f}%)")

        fig, ax = plt.subplots(figsize=figsize)
        bars = sns.barplot(
            x=top_subindustries.values,
            y=top_subindustries.index,
            palette='magma',
            ax=ax
        )

        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            percentage = top_percentages.iloc[i]
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} ({percentage:.1f}%)',
                    ha='left', va='center', fontweight='bold', fontsize=9)

        ax.set_title(f'Top {top_n} GICS Sub-Industries in S&P 500\n({total_unique} total sub-industries)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Companies', fontsize=12)
        ax.set_ylabel('GICS Sub-Industry', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.subplots_adjust(right=0.78)

        return fig, subindustry_summary

    fig_subindustry, subindustry_summary_data = analyze_subindustry_distribution(df.copy())
    st.pyplot(fig_subindustry)



    # ------------ ANALYZE SUB-INDUSTRY DIVERSITY WITHIN SECTORS ------------

    st.header("4. Sub-Industry Diversity Within Sectors")
    def analyze_sector_subindustry_diversity(df_input, sector_col='gics_sector', subindustry_col='gics_sub_industry'):
        sector_diversity = df_input.groupby(sector_col)[subindustry_col].nunique().sort_values(ascending=False)
        st.subheader("Sub-Industry Diversity by Sector:")
        
        # Display as a dataframe for better readability in Streamlit
        st.dataframe(pd.DataFrame(sector_diversity).reset_index().rename(columns={subindustry_col: 'Unique Sub-Industries'}))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=sector_diversity.values, y=sector_diversity.index, palette='crest', ax=ax)
        ax.set_title('Number of Unique Sub-Industries per GICS Sector', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Unique Sub-Industries', fontsize=12)
        ax.set_ylabel('GICS Sector', fontsize=12)
        plt.tight_layout()
        return fig, sector_diversity

    fig_sector_diversity, sector_diversity_data = analyze_sector_subindustry_diversity(df.copy())
    st.pyplot(fig_sector_diversity)

else:
    st.error("Could not load S&P 500 data. Please check your internet connection or 'sp500.csv' file.")