import ollama
import pandas as pd
import os
import re

print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
for file in os.listdir('.'):
    if file.endswith('.csv'):
        print(f"  - {file}")
print("\n" + "="*50 + "\n")

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
    
    # Defining the allowed categories
    allowed_categories = ['Food Quality', 'Service', 'Cleanliness', 'Value', 'Ambiance']
    
    # Split into lines
    lines = feedback_text.strip().split('\n')
    valid_points = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip lines with these phrases
        skip_phrases = [
            'not mentioned', 'not explicitly', 'not specified',
            'implied', 'not discussed', 'none', 'n/a',
            'can be considered', 'however', 'since',
            'although', 'note:', 'the review'
        ]
        
        if any(phrase in line.lower() for phrase in skip_phrases):
            continue
        
        # Check if line starts with a number
        if not re.match(r'^\d+\.', line):
            continue
        
        # Check if it contains an allowed category
        has_valid_category = False
        for category in allowed_categories:
            if category in line:
                has_valid_category = True
                break
        
        if not has_valid_category:
            continue
        
        # Clean up the line (remove parenthetical explanations)
        line = re.sub(r'\(.*?\)', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        
        valid_points.append(line)
    
    # Return only first 4 points
    result = valid_points[:4]
    
    # If we have less than 2 points, return original
    if len(result) < 2:
        return feedback_text
    
    return '\n'.join(result)


def standardize_feedback(feedback_text):
    """Standardize common subcategory variations to consistent terms"""
    
    # Mapping of variations to standard terms
    standardizations = {
        # Service variations
        r'Service:.*?(slow|wait|delay|took.*long|forever|responsiveness|speed)': 'Service: speed',
        r'Service:.*?(friendly|rude|attitude|interaction)': 'Service: friendliness',
        r'Service:.*?(attentive|attention|check)': 'Service: attentiveness',
        r'Service:.*?(professional|accommodation|handling)': 'Service: professionalism',
        
        # Food Quality variations
        r'Food Quality:.*?(cold|warm|hot|temperature)': 'Food Quality: temperature',
        r'Food Quality:.*?(delicious|taste|flavor|yummy)': 'Food Quality: taste',
        r'Food Quality:.*?(fresh|stale)': 'Food Quality: freshness',
        r'Food Quality:.*?(portion|size|amount)': 'Food Quality: portion size',
        r'Food Quality:.*?(variety|options|selection)': 'Food Quality: variety',
        r'Food Quality:.*?(presentation|plating|appearance)': 'Food Quality: presentation',
        
        # Value variations
        r'Value:.*?(expensive|pricey|cheap|cost|price|pricing)': 'Value: pricing',
        r'Value:.*?(worth|money|value)': 'Value: quality for cost',
        
        # Ambiance variations
        r'Ambiance:.*?(loud|quiet|noise)': 'Ambiance: noise level',
        r'Ambiance:.*?(decor|decoration|aesthetic)': 'Ambiance: decor',
        r'Ambiance:.*?(comfort|cozy|space)': 'Ambiance: comfort',
        r'Ambiance:.*?(atmosphere|vibe|ambiance)': 'Ambiance: atmosphere',
        r'Ambiance:.*?(light|lighting|bright|dark)': 'Ambiance: lighting',
        
        # Cleanliness variations
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
                # Renumber sequentially and apply standardization
                standardized_lines.append(f"{len(standardized_lines) + 1}. {replacement}")
                standardized = True
                break
        
        if not standardized and line.strip():
            # Keep original but renumber
            # Extract just the category and detail part (remove old number)
            content = re.sub(r'^\d+\.\s*', '', line)
            if content:
                standardized_lines.append(f"{len(standardized_lines) + 1}. {content}")
    
    return '\n'.join(standardized_lines)


def generate_improvement_suggestion(feedback_point):
    """Generate actionable improvement suggestion for a feedback point"""
    
    prompt = f"""Given this customer feedback about a restaurant, provide ONE specific, actionable improvement suggestion.

Feedback: {feedback_point}

Requirements:
- Be specific and actionable
- Focus on practical business solutions
- Keep it brief (1-2 sentences)
- Make it relevant to restaurant operations

Improvement suggestion:"""
    
    response = ollama.generate(
        model='mistral:latest',
        prompt=prompt,
        options={
            'temperature': 0.3,
            'num_predict': 80,
        }
    )
    
    return response['response'].strip()


def analyze_review_with_suggestions(review_text):
    """
    Complete pipeline: Extract feedback categories AND generate improvement suggestions
    Returns: dict with feedback_points and suggestions
    """
    # Extract and standardize feedback
    raw_feedback = analyze_review(review_text)
    cleaned_feedback = clean_feedback(raw_feedback)
    standardized_feedback = standardize_feedback(cleaned_feedback)
    
    # Generate suggestions for each feedback point
    feedback_lines = [line for line in standardized_feedback.split('\n') if line.strip()]
    suggestions = []
    
    for feedback_point in feedback_lines:
        # Remove the number prefix for suggestion generation
        clean_point = re.sub(r'^\d+\.\s*', '', feedback_point)
        suggestion = generate_improvement_suggestion(clean_point)
        suggestions.append(f"{feedback_point} → {suggestion}")
    
    return {
        'feedback_points': standardized_feedback,
        'suggestions': '\n'.join(suggestions)
    }


def test_full_system():
    """Test the complete system with categorization + suggestions"""
    
    test_reviews = [
        "The pasta was cold when it arrived and the waiter took forever to check on us.",
        "Great food but way too expensive for the portion sizes.",
        "The restaurant was incredibly loud and we could barely hear each other.",
    ]
    
    print("\n" + "="*50)
    print("FULL SYSTEM TEST: Categorization + Suggestions")
    print("="*50 + "\n")
    
    for i, review in enumerate(test_reviews, 1):
        print(f"Test {i}:")
        print(f"Review: \"{review}\"\n")
        
        result = analyze_review_with_suggestions(review)
        
        print("Feedback Categories:")
        print(result['feedback_points'])
        print("\nImprovement Suggestions:")
        print(result['suggestions'])
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    print("Philadelphia Restaurant Review Analysis System\n" + "="*50 + "\n")
    
    # Dataset
    df = pd.read_csv('/Users/Enrique/ALY 6040 Files/philly_food_combined_final.csv')
    
    print("Choose an option:")
    print("1. Test system on sample reviews")
    print("2. Process first 10 reviews from dataset")
    print("3. Process full dataset (760K reviews: takes 2-3 hours)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Test cases
        test_full_system()
    
    elif choice == "2":
        # Process first 10 reviews
        print(f"\nProcessing first 10 reviews from dataset of {len(df)} reviews\n")
        
        for i in range(min(10, len(df))):
            review_text = df.iloc[i]['text']
            
            print(f"Review {i+1}:")
            print(review_text[:100] + "...\n")
            
            result = analyze_review_with_suggestions(review_text)
            
            print("Feedback Categories:")
            print(result['feedback_points'])
            print("\nImprovement Suggestions:")
            print(result['suggestions'])
            print("\n" + "="*50 + "\n")
    
    elif choice == "3":
        # Process full dataset
        print(f"\nProcessing all {len(df)} reviews...")
        print("This will take 2-3 hours. Progress will be shown every 1000 reviews.\n")
        
        feedback_results = []
        suggestion_results = []
        
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"Processed {i}/{len(df)} reviews...")
            
            review_text = row['text']
            result = analyze_review_with_suggestions(review_text)
            
            feedback_results.append(result['feedback_points'])
            suggestion_results.append(result['suggestions'])
        
        # Add results to dataframe
        df['feedback_categories'] = feedback_results
        df['improvement_suggestions'] = suggestion_results
        
        # Save results
        output_file = 'philly_reviews_analyzed_with_suggestions.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n Done! Results saved to: {output_file}")
        
        # Show summary statistics
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        all_feedback = df['feedback_categories'].str.lower()
        category_counts = {
            'Food Quality': all_feedback.str.contains('food quality').sum(),
            'Service': all_feedback.str.contains('service:').sum(),
            'Value': all_feedback.str.contains('value:').sum(),
            'Ambiance': all_feedback.str.contains('ambiance:').sum(),
            'Cleanliness': all_feedback.str.contains('cleanliness:').sum()
        }
        
        print("\nFeedback Category Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(df)) * 100
            print(f"{category}: {count:,} reviews ({percentage:.1f}%)")
    
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

## What This Complete System Does:

### Extracts Feedback Categories
'''
Review: "The pasta was cold and service was slow"
Categories:
1. Food Quality: temperature
2. Service: speed
```

### **2. Generates Improvement Suggestions** ✅
```
Improvement Suggestions:
1. Food Quality: temperature → Implement heated plate warmers and train kitchen staff to minimize time between cooking and serving
2. Service: speed → Analyze peak hours and add staff during busy periods, or streamline the order-taking process
'''