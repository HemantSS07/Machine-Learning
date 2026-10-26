# AI-Powered Career Guidance — Dataset Package

## Files
- `career_guidance_raw_synthetic.csv`: synthetic raw data, includes missing values and 12 intentionally injected duplicate rows.
- `career_guidance_processed.csv`: duplicates removed, numeric missing values imputed with medians, numeric assessment scores normalized to [0, 1].
- `career_guidance_train.csv`: 80% training split from cleaned raw data, with original 1–10 score scale.
- `career_guidance_test.csv`: 20% test split from cleaned raw data, with original 1–10 score scale.
- `data_dictionary.json`: feature definitions and limitations.

## Dataset design
- Intended for prototyping an ML career-domain classifier/recommender.
- 12 target domains.
- 24 assessment features (6 interests, 5 aptitude indicators, 5 personality-trait indicators, 5 skills, plus 4 profile/context categorical fields).
- `profile_id` is an identifier and must not be used as a predictive feature.

## Important methodological note
This is **synthetic, simulated data** generated from hand-designed career-domain templates. It is useful for testing data preprocessing, pipelines, model code, API integration, and UI flows. It is NOT real-world evidence that psychometric scores predict career outcomes. Do not present model accuracy on this data as real-world career-guidance accuracy.

## Suggested next steps
1. Run EDA and check class balance.
2. Build a preprocessing pipeline (median imputation + scaling for numeric data, most-frequent imputation + one-hot encoding for categorical data).
3. Compare a DummyClassifier baseline, Logistic Regression, Random Forest, and KNN using stratified cross-validation.
4. Report macro-F1, balanced accuracy, confusion matrix, and top-3 accuracy.
5. Use a clearly labeled demo and gather consented, anonymized, ethically collected real data before making real-world claims.
