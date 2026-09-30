# Trial matches for TCGA-05-4418-derived

Generated 2026-09-30T07:05:08+00:00 · prompt `3fe32ef28a11` · local `phi4` · API `disabled`

> Research prototype. Not for clinical use.

```
PATIENT PROFILE (as of 2026-09-30)
Diagnosis: Lung adenocarcinoma, stage IV (histology: adenocarcinoma)
Age 63, male. ECOG 1. Smoking: former. Brain metastases: no. Prior malignancy: not recorded.
Mutations: KRAS p.G12C (hotspot); STK11 p.G56V; KEAP1 p.F280Y; RBM10 p.G17Wfs*4
TMB 4.6 mut/Mb; PD-L1 TPS 15%; MSI: MSS
Treatment history (complete):
  Line 1: carboplatin + pemetrexed + pembrolizumab, 2025-11-10 to 2026-06-20, best response PD, stopped for progression
Next line of therapy: 2
```

3 trials, 43 criteria · API routing rate 0.0 · unknown criteria 19 · unresolved trials 2

| Rank | NCT ID | Status | Score | Retrieval | Pass/Fail/Unknown | Title |
|---:|---|---|---:|---:|---|---|
| 1 | [NCT04303780](https://clinicaltrials.gov/study/NCT04303780) | unresolved | 0.863 | 0.0328 | 9/0/4 | Adagrasib in Previously Treated Patients With NSCLC and KRAS G12C Mutation (KRYSTAL-1) |
| 2 | [NCT03600883](https://clinicaltrials.gov/study/NCT03600883) | unresolved | 0.732 | 0.0320 | 7/0/10 | Sotorasib in Advanced Solid Tumors With KRAS G12C Mutation (CodeBreaK 100) |
| 3 | [NCT03778229](https://clinicaltrials.gov/study/NCT03778229) | ineligible | 0.588 | 0.0320 | 6/2/5 | Osimertinib in EGFR-Mutant NSCLC With Acquired Resistance |

## 1. NCT04303780 — unresolved (score 0.863)

Adagrasib in Previously Treated Patients With NSCLC and KRAS G12C Mutation (KRYSTAL-1)

| Kind | # | Criterion | Verdict | Conf | Passes | Route | Evidence |
|---|---:|---|---|---:|---|---|---|
| inc | 0 | Histologically confirmed diagnosis of unresectable or metastatic NSCLC. | met | 1.00 | ✓ | local | Diagnosis: Lung adenocarcinoma, stage IV |
| inc | 1 | Presence of KRAS G12C mutation in tumor tissue. | met | 1.00 | ✓ | local | mutations: KRAS p.G12C |
| inc | 2 | Prior treatment with at least one systemic anti-cancer therapy for advanced disease including an anti-PD-1/PD-L1 antibody (unless contraindi | met | 0.95 | ✓ | local | treatment_history line 1: pembrolizumab |
| inc | 3 | Adequate organ function. | unknown | 0.90 | ? | local | none |
| inc | 4 | ECOG performance status of 0 or 1. | met | 1.00 | ✓ | local | ECOG 1 |
| inc | 5 | Life expectancy of at least 3 months. | unknown | 0.70 | ? | api_unavailable | none |
| inc | 6 | Age >= 18 years. | met | 1.00 | ✓ | local | age 63 |
| exc | 0 | Active brain metastases (patients with treated, stable brain metastases are eligible). | not met | 1.00 | ✓ | local | brain metastases: no |
| exc | 1 | Prior treatment with a KRAS G12C-specific inhibitor. | not met | 1.00 | ✓ | local | treatment_history line 1: carboplatin + pemetrexed + pembrolizumab |
| exc | 2 | Recent major surgery within 4 weeks. | not met | 0.90 | ✓ | local | none |
| exc | 3 | History of intestinal disease or major gastric surgery that would impair drug absorption. | unknown | 0.90 | ? | local | none |
| exc | 4 | Serious cardiac illness including congestive heart failure NYHA class III or IV. | unknown | 0.90 | ? | local | none |
| exc | 5 | Pregnant or breastfeeding. | not met | 1.00 | ✓ | local | gender: male |

## 2. NCT03600883 — unresolved (score 0.732)

Sotorasib in Advanced Solid Tumors With KRAS G12C Mutation (CodeBreaK 100)

| Kind | # | Criterion | Verdict | Conf | Passes | Route | Evidence |
|---|---:|---|---|---:|---|---|---|
| inc | 0 | Subject has provided informed consent prior to initiation of any study specific activities/procedures. | unknown | 0.90 | ? | local | none |
| inc | 1 | Male or female, aged >=18 years. | met | 1.00 | ✓ | local | age 63, male |
| inc | 2 | Pathologically documented, locally-advanced or metastatic malignancy with KRAS p.G12C mutation identified through molecular testing. | met | 1.00 | ✓ | local | diagnosis: Lung adenocarcinoma, stage IV; mutations: KRAS p.G12C |
| inc | 3 | Subjects with active brain metastases from non-brain tumors are eligible if they have been treated and are without evidence of progression o | met | 1.00 | ✓ | local | brain_metastases: no |
| inc | 4 | Measurable disease per Response Evaluation Criteria in Solid Tumors 1.1 (RECIST 1.1) criteria. | unknown | 0.70 | ? | api_unavailable | none |
| inc | 5 | Eastern Cooperative Oncology Group (ECOG) Performance Status of <=2. | met | 1.00 | ✓ | local | ECOG 1 |
| inc | 6 | Life expectancy of >3 months, in the opinion of the investigator. | unknown | 0.90 | ? | local | none |
| inc | 7 | Adequate organ function. | unknown | 0.90 | ? | local | none |
| exc | 0 | Active infection requiring systemic therapy. | unknown | 0.90 | ? | local | none |
| exc | 1 | Known positive test for human immunodeficiency virus, hepatitis C virus, chronic active hepatitis B infection. | unknown | 1.00 | ? | local | none |
| exc | 2 | Myocardial infarction within 6 months of study day 1, symptomatic congestive heart failure (New York Heart Association > class II), unstable | unknown | 0.90 | ? | local | none |
| exc | 3 | Gastrointestinal (GI) tract disease causing the inability to take oral medication, malabsorption syndrome, requirement for intravenous alime | unknown | 0.90 | ? | local | none |
| exc | 4 | Prior therapy with a direct and selective KRAS G12C inhibitor. | not met | 1.00 | ✓ | local | treatment_history line 1: carboplatin + pemetrexed + pembrolizumab |
| exc | 5 | Anti-tumor therapy within 28 days of study day 1. | unknown | 0.90 | ? | local | none |
| exc | 6 | Therapeutic or palliative radiation therapy within 4 weeks of study day 1. | unknown | 0.90 | ? | local | none |
| exc | 7 | Currently receiving treatment in another investigational device or drug study. | not met | 0.90 | ✓ | local | treatment_history next line of therapy: 2 |
| exc | 8 | Female subjects of childbearing potential unwilling to use an acceptable method(s) of effective contraception during treatment. | not met | 1.00 | ✓ | local | gender: male |

## 3. NCT03778229 — ineligible (score 0.588)

Osimertinib in EGFR-Mutant NSCLC With Acquired Resistance

**Blocking criteria**
- *inclusion #1*: Documented EGFR mutation known to be associated with EGFR TKI sensitivity (including exon 19 deletion, L858R, T790M).  
  → The patient has a KRAS p.G12C mutation, not an EGFR mutation associated with TKI sensitivity. (evidence: mutations: KRAS p.G12C)
- *inclusion #2*: Prior treatment with a first- or second-generation EGFR TKI.  
  → The patient's treatment history does not include any first- or second-generation EGFR TKI. (evidence: treatment_history line 1: carboplatin + pemetrexed + pembrolizumab)

| Kind | # | Criterion | Verdict | Conf | Passes | Route | Evidence |
|---|---:|---|---|---:|---|---|---|
| inc | 0 | Histologically or cytologically confirmed NSCLC. | met | 1.00 | ✓ | local | Diagnosis: Lung adenocarcinoma, stage IV |
| inc | 1 | Documented EGFR mutation known to be associated with EGFR TKI sensitivity (including exon 19 deletion, L858R, T790M). | not met | 1.00 | ✗ | local | mutations: KRAS p.G12C |
| inc | 2 | Prior treatment with a first- or second-generation EGFR TKI. | not met | 1.00 | ✗ | local | treatment_history line 1: carboplatin + pemetrexed + pembrolizumab |
| inc | 3 | Radiographic progression on prior EGFR TKI therapy. | unknown | 0.90 | ? | local | treatment_history line 1: carboplatin + pemetrexed + pembrolizumab |
| inc | 4 | Age >= 18 years. | met | 1.00 | ✓ | local | age 63 |
| inc | 5 | ECOG performance status 0-2. | met | 1.00 | ✓ | local | ECOG 1 |
| inc | 6 | Measurable disease per RECIST v1.1. | unknown | 0.70 | ? | api_unavailable | none |
| inc | 7 | Adequate hematologic, hepatic, and renal function. | unknown | 0.90 | ? | local | none |
| exc | 0 | Symptomatic CNS metastases requiring immediate local therapy. | not met | 1.00 | ✓ | local | brain metastases: no |
| exc | 1 | Interstitial lung disease or pneumonitis. | unknown | 0.90 | ? | local | none |
| exc | 2 | Prior treatment with osimertinib or other third-generation EGFR TKI. | not met | 1.00 | ✓ | local | treatment_history line 1: carboplatin + pemetrexed + pembrolizumab |
| exc | 3 | Concurrent malignancy requiring active therapy. | not met | 1.00 | ✓ | local | prior_malignancy: not recorded |
| exc | 4 | Serious cardiac disease including QTc > 470 ms. | unknown | 0.90 | ? | local | none |
