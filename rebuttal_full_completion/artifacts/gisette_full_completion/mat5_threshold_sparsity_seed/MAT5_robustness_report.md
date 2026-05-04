# MAT-5 Threshold/Sparsity/Seed Robustness

- Generated at: 2026-05-03T15:29:44.469319+00:00
- Source root: `/home/harp3133t/Research/03_Research/GH-ANFIS_E404`
- Weight root: `/home/harp3133t/Research/03_Research/GH-ANFIS_E404/hyper_parameter/cv_weights`
- CV split: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Tau sweep: [0.3, 0.4, 0.5, 0.6, 0.7]
- Tau sweep uses existing no_mi GH-ANFIS checkpoints for all four submitted datasets.
- Lambda/seed sweep: dataset=Gisette, fold=1 fixed split, lambda_base_s1=lambda_resid_s2 in [0.0, 0.01, 0.1], seeds=[0, 1, 2, 3, 4].

## Tau 0.5 Summary
| dataset | full_acc_mean | full_acc_std | full_selected_mean | base_selected_mean | residual_effective_selected_mean | full_mask_jaccard_mean | primary_complementary_overlap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Breast_Cancer_Wisconsin_(Original) | 0.9743 | 0.0057 | 9.0000 | 6.6000 | 2.4000 | 1.0000 | 0.0000 |
| Gisette | 0.9758 | 0.0042 | 2680.8000 | 2408.8000 | 272.0000 | 0.6011 | 0.0000 |
| Spambase | 0.9322 | 0.0041 | 50.2000 | 25.2000 | 25.0000 | 0.8490 | 0.0000 |
| Vowel | 0.9667 | 0.0158 | 26.0000 | 26.0000 | 0.0000 | 1.0000 | 0.0000 |

## Tau Range
| dataset | tau | full_acc_mean | full_selected_mean | full_mask_jaccard_mean | primary_complementary_overlap_mean |
| --- | --- | --- | --- | --- | --- |
| Breast_Cancer_Wisconsin_(Original) | 0.3000 | 0.9685 | 9.0000 | 1.0000 | 0.0000 |
| Breast_Cancer_Wisconsin_(Original) | 0.4000 | 0.9728 | 9.0000 | 1.0000 | 0.0000 |
| Breast_Cancer_Wisconsin_(Original) | 0.5000 | 0.9743 | 9.0000 | 1.0000 | 0.0000 |
| Breast_Cancer_Wisconsin_(Original) | 0.6000 | 0.9700 | 8.2000 | 0.8222 | 0.0000 |
| Breast_Cancer_Wisconsin_(Original) | 0.7000 | 0.9571 | 7.4000 | 0.7222 | 0.0000 |
| Gisette | 0.3000 | 0.9757 | 5000.0000 | 1.0000 | 0.0000 |
| Gisette | 0.4000 | 0.9757 | 5000.0000 | 1.0000 | 0.0000 |
| Gisette | 0.5000 | 0.9758 | 2680.8000 | 0.6011 | 0.0000 |
| Gisette | 0.6000 | 0.5000 | 0.0000 | 1.0000 | 0.0000 |
| Gisette | 0.7000 | 0.5000 | 0.0000 | 1.0000 | 0.0000 |
| Spambase | 0.3000 | 0.9211 | 57.0000 | 1.0000 | 0.0000 |
| Spambase | 0.4000 | 0.9248 | 56.8000 | 0.9930 | 0.0000 |
| Spambase | 0.5000 | 0.9322 | 50.2000 | 0.8490 | 0.0000 |
| Spambase | 0.6000 | 0.9263 | 28.6000 | 0.7034 | 0.0000 |
| Spambase | 0.7000 | 0.8683 | 18.0000 | 0.7761 | 0.0000 |
| Vowel | 0.3000 | 0.9667 | 26.0000 | 1.0000 | 0.0000 |
| Vowel | 0.4000 | 0.9667 | 26.0000 | 1.0000 | 0.0000 |
| Vowel | 0.5000 | 0.9667 | 26.0000 | 1.0000 | 0.0000 |
| Vowel | 0.6000 | 0.9667 | 26.0000 | 1.0000 | 0.0000 |
| Vowel | 0.7000 | 0.9626 | 25.6000 | 0.9769 | 0.0000 |

## Lambda/Seed Sweep
| dataset | lambda_sparsity | full_acc_mean | full_acc_std | full_selected_mean | full_selected_std | full_mask_jaccard_mean | primary_complementary_overlap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gisette | 0.0000 | 0.9755 | 0.0015 | 4466.8000 | 17.4402 | 0.8538 | 0.0000 |
| Gisette | 0.0100 | 0.9753 | 0.0022 | 2416.4000 | 46.5300 | 0.7794 | 0.0000 |
| Gisette | 0.1000 | 0.9700 | 0.0022 | 332.2000 | 2.3152 | 0.8441 | 0.0000 |

## Interpretation
- Primary/Complementary effective overlap is expected to be 0 because residual routing uses the hard complement of the primary mask.
- Treat these as robustness artifacts, not replacements for the February submission performance tables.