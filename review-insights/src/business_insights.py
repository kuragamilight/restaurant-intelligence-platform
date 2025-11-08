import ollama
import pandas as pd
import os
import re
from collections import Counter

print("="*70)
print("BUSINESS-LEVEL INSIGHTS ANALYZER")
print("Aggregated Analysis Across All Reviews for a Business")
print("="*70 + "\n")

def analyze_review(review_text):
    """Extract 3-4 standardized feedback points from a review"""
    
    prompt = f"""Extract 3-4 key feedback points from this review.

Use ONLY these categories:
- Food Quality
- Service
- Cleanliness
- Value
- Ambiance

Format: "Category: detail"

Examples:
"Food was cold and service slow" → 1. Food Quality: temperature 2. Service: speed
"Great atmosphere but pricey" → 1. Ambiance: atmosphere 2. Value: pricing

Review: "{review_text}"

Feedback points:"""
    
    response = ollama.generate(
        model='mistral:latest',
        prompt=prompt,
        options={
            'temperature': 0.1,
            'top_p': 0.85,
            'num_predict': 100,
        }
    )
    
    return response['response']


def clean_feedback(feedback_text):
    """Clean and validate feedback points"""
    
    allowed_categories = ['Food Quality', 'Service', 'Cleanliness', 'Value', 'Ambiance']
    lines = feedback_text.strip().split('\n')
    valid_points = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        skip_phrases = [
            'not mentioned', 'not explicitly', 'not specified',
            'implied', 'not discussed', 'none', 'n/a',
            'can be considered', 'however', 'since',
            'although', 'note:', 'the review'
        ]
        
        if any(phrase in line.lower() for phrase in skip_phrases):
            continue
        
        if not re.match(r'^\d+\.', line):
            continue
        
        has_valid_category = False
        for category in allowed_categories:
            if category in line:
                has_valid_category = True
                break
        
        if not has_valid_category:
            continue
        
        line = re.sub(r'\(.*?\)', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        
        valid_points.append(line)
    
    result = valid_points[:4]
    
    if len(result) < 2:
        return feedback_text
    
    return '\n'.join(result)


def standardize_feedback(feedback_text):
    """Standardize common subcategory variations to consistent terms"""
    
    standardizations = {
        r'Service:.*?(slow|wait|delay|took.*long|forever|responsiveness|speed)': 'Service: speed',
        r'Service:.*?(friendly|rude|attitude|interaction)': 'Service: friendliness',
        r'Service:.*?(attentive|attention|check)': 'Service: attentiveness',
        r'Service:.*?(professional|accommodation|handling)': 'Service: professionalism',
        r'Food Quality:.*?(cold|warm|hot|temperature)': 'Food Quality: temperature',
        r'Food Quality:.*?(delicious|taste|flavor|yummy)': 'Food Quality: taste',
        r'Food Quality:.*?(fresh|stale)': 'Food Quality: freshness',
        r'Food Quality:.*?(portion|size|amount)': 'Food Quality: portion size',
        r'Food Quality:.*?(variety|options|selection)': 'Food Quality: variety',
        r'Food Quality:.*?(presentation|plating|appearance)': 'Food Quality: presentation',
        r'Value:.*?(expensive|pricey|cheap|cost|price|pricing)': 'Value: pricing',
        r'Value:.*?(worth|money|value)': 'Value: quality for cost',
        r'Ambiance:.*?(loud|quiet|noise)': 'Ambiance: noise level',
        r'Ambiance:.*?(decor|decoration|aesthetic)': 'Ambiance: decor',
        r'Ambiance:.*?(comfort|cozy|space)': 'Ambiance: comfort',
        r'Ambiance:.*?(atmosphere|vibe|ambiance)': 'Ambiance: atmosphere',
        r'Ambiance:.*?(light|lighting|bright|dark)': 'Ambiance: lighting',
        r'Cleanliness:.*?(clean|dirty|hygiene|sanitary)': 'Cleanliness: overall hygiene',
    }
    
    lines = feedback_text.split('\n')
    standardized_lines = []
    
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
            
        standardized = False
        for pattern, replacement in standardizations.items():
            if re.search(pattern, line, re.IGNORECASE):
                standardized_lines.append(f"{len(standardized_lines) + 1}. {replacement}")
                standardized = True
                break
        
        if not standardized and line.strip():
            content = re.sub(r'^\d+\.\s*', '', line)
            if content:
                standardized_lines.append(f"{len(standardized_lines) + 1}. {content}")
    
    return '\n'.join(standardized_lines)


def extract_feedback_categories(feedback_text):
    """Extract just the category labels from feedback text for aggregation"""
    categories = []
    lines = feedback_text.split('\n')
    
    for line in lines:
        line = re.sub(r'^\d+\.\s*', '', line.strip())
        if line:
            categories.append(line)
    
    return categories


def generate_business_improvement(issue, mention_count, total_reviews):
    """Generate improvement suggestion based on aggregated data"""
    
    percentage = (mention_count / total_reviews) * 100
    
    prompt = f"""A restaurant has received customer feedback about: "{issue}"

Context:
- This issue was mentioned in {mention_count} out of {total_reviews} reviews ({percentage:.1f}%)
- This is a recurring theme across multiple customers

Provide ONE specific, actionable improvement recommendation that addresses the root cause of this issue. Focus on practical solutions the business can implement.

Recommendation:"""
    
    response = ollama.generate(
        model='mistral:latest',
        prompt=prompt,
        options={
            'temperature': 0.3,
            'num_predict': 100,
        }
    )
    
    return response['response'].strip()


def analyze_business_reviews(df, business_id=None, business_name=None):
    """
    Analyze all reviews for a specific business and generate aggregated insights
    """
    
    # Filter reviews for the specific business
    if business_id:
        business_reviews = df[df['business_id'] == business_id].copy()
        identifier = f"Business ID: {business_id}"
    elif business_name:
        business_reviews = df[df['name'].str.contains(business_name, case=False, na=False)].copy()
        if len(business_reviews) > 0:
            identifier = f"Business: {business_reviews.iloc[0]['name']}"
            actual_business_id = business_reviews.iloc[0]['business_id']
        else:
            identifier = f"Business: {business_name}"
    else:
        print("Error: Must provide either business_id or business_name")
        return None
    
    if len(business_reviews) == 0:
        print(f"No reviews found for {identifier}")
        return None
    
    print(f"\n{'='*70}")
    print(f"BUSINESS ANALYSIS: {identifier}")
    print(f"{'='*70}")
    print(f"Total Reviews: {len(business_reviews)}")
    print(f"Average Rating: {business_reviews['stars_review'].mean():.2f} stars")
    print(f"\nAnalyzing reviews (this may take a few minutes)...\n")
    
    # Analyze each review and collect feedback categories
    all_feedback_categories = []
    processed_count = 0
    
    for idx, row in business_reviews.iterrows():
        review_text = row['text']
        
        # Extract feedback
        raw_feedback = analyze_review(review_text)
        cleaned_feedback = clean_feedback(raw_feedback)
        standardized_feedback = standardize_feedback(cleaned_feedback)
        
        # Extract categories for aggregation
        categories = extract_feedback_categories(standardized_feedback)
        all_feedback_categories.extend(categories)
        
        # Show progress
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count}/{len(business_reviews)} reviews...")
    
    print(f"  Processed {processed_count}/{len(business_reviews)} reviews... Done!\n")
    
    # Count frequency of each feedback category
    category_counts = Counter(all_feedback_categories)
    
    # Get top issues (most common feedback)
    top_issues = category_counts.most_common(10)
    
    print(f"{'='*70}")
    print("TOP 10 CUSTOMER FEEDBACK THEMES")
    print(f"{'='*70}\n")
    
    for i, (issue, count) in enumerate(top_issues, 1):
        percentage = (count / len(business_reviews)) * 100
        print(f"{i}. {issue}")
        print(f"   Mentioned in {count} reviews ({percentage:.1f}% of reviews)\n")
    
    # Generate aggregated improvement suggestions
    print(f"{'='*70}")
    print("PRIORITY IMPROVEMENT RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    print("Generating recommendations based on top issues...\n")
    
    for i, (issue, count) in enumerate(top_issues[:5], 1):  # Top 5 issues
        percentage = (count / len(business_reviews)) * 100
        suggestion = generate_business_improvement(issue, count, len(business_reviews))
        
        priority = 'HIGH' if percentage > 20 else 'MEDIUM' if percentage > 10 else 'LOW'
        
        print(f"{i}. {issue} ({percentage:.1f}% of reviews)")
        print(f"   Priority: {priority}")
        print(f"   Recommendation: {suggestion}\n")
    
    # Category-level summary
    print(f"{'='*70}")
    print("FEEDBACK CATEGORY BREAKDOWN")
    print(f"{'='*70}\n")
    
    category_summary = {
        'Food Quality': 0,
        'Service': 0,
        'Cleanliness': 0,
        'Value': 0,
        'Ambiance': 0
    }
    
    for category, count in category_counts.items():
        for main_cat in category_summary.keys():
            if category.startswith(main_cat):
                category_summary[main_cat] += count
                break
    
    total_mentions = sum(category_summary.values())
    
    for category, count in sorted(category_summary.items(), key=lambda x: x[1], reverse=True):
        if total_mentions > 0:
            percentage = (count / total_mentions) * 100
            print(f"{category}: {count} mentions ({percentage:.1f}% of total feedback)")
    
    # Strengths and weaknesses
    print(f"\n{'='*70}")
    print("RATING DISTRIBUTION")
    print(f"{'='*70}\n")
    
    positive_reviews = business_reviews[business_reviews['stars_review'] >= 4]
    negative_reviews = business_reviews[business_reviews['stars_review'] <= 2]
    neutral_reviews = business_reviews[business_reviews['stars_review'] == 3]
    
    print(f"Positive Reviews (4-5 stars): {len(positive_reviews)} ({len(positive_reviews)/len(business_reviews)*100:.1f}%)")
    print(f"Negative Reviews (1-2 stars): {len(negative_reviews)} ({len(negative_reviews)/len(business_reviews)*100:.1f}%)")
    print(f"Neutral Reviews (3 stars): {len(neutral_reviews)} ({len(neutral_reviews)/len(business_reviews)*100:.1f}%)")
    
    return {
        'business_identifier': identifier,
        'total_reviews': len(business_reviews),
        'avg_rating': business_reviews['stars_review'].mean(),
        'top_issues': top_issues,
        'category_summary': category_summary
    }


def search_businesses(df, search_term):
    """Search for businesses by name"""
    
    matches = df[df['name'].str.contains(search_term, case=False, na=False)][['name', 'business_id']].drop_duplicates()
    
    if len(matches) == 0:
        print(f"No businesses found matching '{search_term}'")
        return None
    
    print(f"\nFound {len(matches)} business(es) matching '{search_term}':\n")
    
    for i, (idx, row) in enumerate(matches.head(20).iterrows(), 1):
        review_count = len(df[df['business_id'] == row['business_id']])
        print(f"{i}. {row['name']}")
        print(f"   Business ID: {row['business_id']}")
        print(f"   Reviews: {review_count}\n")
    
    if len(matches) > 20:
        print(f"... and {len(matches) - 20} more matches\n")
    
    return matches


def process_all_businesses(df, output_file='business_insights_summary.csv'):
    """Process all businesses and save summary to CSV"""
    
    unique_businesses = df['business_id'].unique()
    print(f"\nProcessing {len(unique_businesses)} unique businesses...")
    print("This will take several hours.\n")
    
    all_results = []
    
    for i, biz_id in enumerate(unique_businesses, 1):
        if i % 50 == 0:
            print(f"\nProcessed {i}/{len(unique_businesses)} businesses...")
        
        business_reviews = df[df['business_id'] == biz_id]
        business_name = business_reviews.iloc[0]['name'] if 'name' in business_reviews.columns else 'Unknown'
        
        # Collect feedback categories
        all_feedback = []
        for _, row in business_reviews.iterrows():
            raw = analyze_review(row['text'])
            cleaned = clean_feedback(raw)
            standardized = standardize_feedback(cleaned)
            categories = extract_feedback_categories(standardized)
            all_feedback.extend(categories)
        
        # Count categories
        category_counts = Counter(all_feedback)
        top_5 = category_counts.most_common(5)
        
        all_results.append({
            'business_id': biz_id,
            'business_name': business_name,
            'total_reviews': len(business_reviews),
            'avg_rating': business_reviews['stars_review'].mean(),
            'top_issue_1': top_5[0][0] if len(top_5) > 0 else '',
            'top_issue_1_count': top_5[0][1] if len(top_5) > 0 else 0,
            'top_issue_2': top_5[1][0] if len(top_5) > 1 else '',
            'top_issue_2_count': top_5[1][1] if len(top_5) > 1 else 0,
            'top_issue_3': top_5[2][0] if len(top_5) > 2 else '',
            'top_issue_3_count': top_5[2][1] if len(top_5) > 2 else 0,
        })
    
    # Save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Results saved to: {output_file}")


if __name__ == "__main__":
    
    # Load data
    df = pd.read_csv('/Users/Enrique/ALY 6040 Files/philly_food_combined_final.csv')
    
    print(f"Loaded {len(df):,} reviews from {df['business_id'].nunique():,} unique businesses\n")
    
    print("Choose an option:")
    print("1. Search for a business by name")
    print("2. Analyze business by exact name")
    print("3. Analyze business by business_id")
    print("4. Process ALL businesses and save summary CSV (takes several hours)")
    
    choice = input("\nEnter choice (1/2/3/4): ").strip()
    
    if choice == "1":
        search_term = input("\nEnter business name to search: ").strip()
        matches = search_businesses(df, search_term)
        
        if matches is not None and len(matches) > 0:
            analyze = input("\nAnalyze one of these businesses? (y/n): ").strip().lower()
            if analyze == 'y':
                business_name = input("Enter exact business name from list above: ").strip()
                analyze_business_reviews(df, business_name=business_name)
    
    elif choice == "2":
        business_name = input("\nEnter exact business name: ").strip()
        analyze_business_reviews(df, business_name=business_name)
    
    elif choice == "3":
        business_id = input("\nEnter business_id: ").strip()
        analyze_business_reviews(df, business_id=business_id)
    
    elif choice == "4":
        confirm = input("\nThis will take several hours. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            output_file = input("Enter output filename (default: business_insights_summary.csv): ").strip()
            if not output_file:
                output_file = 'business_insights_summary.csv'
            process_all_businesses(df, output_file)
        else:
            print("Cancelled.")
    
    else:
        print("Invalid choice.")