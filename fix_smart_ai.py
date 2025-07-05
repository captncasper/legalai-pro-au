import numpy as np
# Add this to smart_legal_ai.py after line 60 (in predict_case_outcome function)

# BETTER pattern matching - more flexible
positive_patterns = [
    (['no warning', 'without warning', 'no written warning'], 20, 'No Written Warnings'),
    (['good performance', 'excellent performance', 'positive review'], 10, 'Good Performance History'),
    (['long service', 'years service', 'years of service'], 10, 'Long Service'),
    (['no opportunity', 'not given chance', 'no chance to respond'], 25, 'No Opportunity to Respond'),
    (['unfair', 'unjust', 'unreasonable'], 15, 'Unfair Treatment'),
    (['discrimination', 'harass', 'bullying'], 20, 'Discrimination/Harassment'),
    (['no policy', 'unwritten rule', 'not in policy'], 15, 'No Written Policy'),
    (['others do it', 'other employees', 'common practice'], 15, 'Inconsistent Enforcement')
]

# Check each pattern group
for patterns, weight, label in positive_patterns:
    if any(pattern in case_lower for pattern in patterns):
        score += weight
        factors['positive'] += 1
        reasoning.append(f"✓ {label} (+{weight}%)")
