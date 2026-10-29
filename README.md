# Self-reported versus measured body mass index and eligibility for obesity treatment in US adults (NHANES)

Analysis code for the study of how often eligibility for anti-obesity medication (BMI >= 30 kg/m², or >= 27 kg/m² with a
weight-related condition) and for metabolic and bariatric surgery (BMI >= 35 kg/m², or >= 30 kg/m² with a metabolic
condition) differs when it is judged from self-reported rather than measured height and weight, in adults in the
National Health and Nutrition Examination Survey (NHANES) 2013-2014, 2015-2016, 2017-2018 and August 2021-August 2023.
It also evaluates measuring height and weight only near the thresholds, statistical correction of self-report, and
differences between groups.

## Running

```
pip install -r requirements.txt
python3 fetch_data.py            # downloads the 38 NHANES files into data/ and checks their MD5
cd code
python3 analysis.py              # sample, rules, bootstrap weights; base estimates   -> out/*.csv, out/flow.json
python3 analysis2.py             # Tables 1-2, standardised subgroups, grey zone, cycles -> out/r2_*
python3 analysis3.py             # correction model, measurement frontier, decomposition -> out/r3_*
python3 analysis4.py             # population counts, paired strategy comparisons, equity -> out/r4_*
python3 analysis5.py             # equity gaps, race-free model, targeted over-reporting -> out/r5_*
python3 analysis6.py             # over-reporting beyond a public rule, error by race    -> out/r6_*
python3 analysis7.py             # further sensitivity analyses, variance check          -> out/r7_*
python3 figs.py                  # Figures 1-3 and S1-S4 (PNG, 700 dpi)                   -> out/
python3 book.py                  # Tables_and_Figures.docx, PNGs and Source_data.xlsx    -> $DELIVERABLE_DIR
```

Run the scripts in this order; analysis2.py to analysis7.py re-execute the data preparation part of analysis.py, and
figs.py and book.py read the files in `out/`. `DELIVERABLE_DIR` defaults to `deliverable/` in the repository.
book_base.py holds the Word layout helpers used by book.py. Confidence intervals come from 1,000 Rao-Wu rescaling
bootstrap resamples of primary sampling units within strata; all random steps use fixed seeds. Comments in the code
are partly in Chinese.

## Data

NHANES public-use files are produced by the National Center for Health Statistics and are available at
https://wwwn.cdc.gov/nchs/nhanes/. They are not redistributed here; `fetch_data.py` downloads them (demographics,
body measures, weight history, blood pressure and cholesterol, diabetes, medical conditions, blood pressure
examination, total cholesterol, glycohemoglobin, and sleep disorders for 2015-2018).

## Software

Python 3.12.3; package versions in `requirements.txt`.

The analysis was written and run by the author in Python 3, with Claude (Anthropic) and ChatGPT (OpenAI) used as
assistants under the author's direction; the author reviewed and tested all code and takes full responsibility for
the results it produces.

## Citation

Yen-Hsiang Wang. Analysis code for "Self-Reported Versus Measured Body Mass Index and Eligibility for Obesity Treatment
in US Adults". https://github.com/Rogerking928/self-reported-bmi-treatment-eligibility

## Licence

MIT.
