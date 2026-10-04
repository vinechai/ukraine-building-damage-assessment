# ukraine building damage assessment

building damage detection for ukraine using sentinel-1 radar satellite data, real building footprints, and real war damage labels from the united nations.

## why not the usual xview2/xbd approach

the standard way to do this kind of project is xview2: train a model on xbd, a dataset of before/after satellite photos with hand drawn building outlines and damage labels. problem is xbd only has natural disasters (hurricanes, earthquakes, wildfires), zero war damage. and there's no free high resolution satellite imagery for ukraine to build an xbd style dataset for the war either (checked maxar's free imagery program directly, it only covers natural disasters, not conflicts).

what real researchers and journalists (including the economist) actually use for ukraine is a different approach. instead of teaching a model to spot buildings in a photo, you start from building locations you already know, and check how the radar signal at each one changed over time. this only needs sentinel-1 satellite data, which is free and public, and radar works through clouds and at night, good for a warzone.

## the three data sources

any damage model needs 3 things: where the buildings are, what the satellite saw there over time, and for at least some buildings, ground truth on what actually happened. we get each one from a different place:

- **building locations**: [microsoft building footprints](https://github.com/microsoft/GlobalMLBuildingFootprints), a free global dataset of building outlines, extracted automatically from satellite imagery. no idea about damage, just shapes.
- **satellite signal**: sentinel-1 radar, free via google earth engine.
- **ground truth**: unosat (un satellite centre) manually looked at real high resolution photos and marked every damaged building they could see, for specific ukrainian cities. real human judgement, not automated, and not related to sentinel/radar at all.

## the benchmark dataset

[pwtt](https://github.com/oballinger/PWTT) is a published, peer reviewed open source project that already combined all 3 sources above for 19 ukrainian cities (plus a few in gaza, syria, iraq). unosat's damage points joined onto microsoft's building footprints, plus some sentinel-1 change detection stats already computed per building. downloaded into `data/benchmark/`, real data, not synthetic.

checked against their published table: mariupol has 18446 buildings, 31.42% damaged, matches exactly. that confirms the loader reads the data correctly.

one thing to know if you look at the data yourself: pwtt's "city" areas aren't always one single shape. mariupol for example is defined as 3 separate disconnected polygons in their code, not the whole city as one rectangle. so a map of mariupol buildings looks like 2 to 3 separate clusters with gaps between them. that's intentional on their part, not a bug in our loading.

this project is a geospatial time series data science project, not a computer vision one. the raw sentinel-1 data is just 2 numbers (VV, VH radar readings) per building per date, no images or pixel grids involved. buildings are too small relative to sentinel-1's 10m resolution for image based approaches to make sense here. no neural network is used, the models are plain scikit-learn classifiers (logistic regression, random forest, gradient boosting) trained on hand engineered features from the raw signal.

there are two clearly separate parts below, kept apart on purpose. part 1 is the finished, validated core project. part 2 is a secondary investigation into whether this extends to present day prediction, and it honestly does not fully work. kept in because the investigation itself (two real bugs found and fixed, one not, and why) is a more useful and more honest thing to show than hiding it or pretending it's settled.

## part 1: the core project, validated against real 2022 ground truth

pwtt's damage detection is a simple statistics rule, no training involved. flag a building damaged if its radar change score is above a threshold. we trained an actual model instead, using the raw sentinel-1 signal per building (not just pwtt's precomputed summary numbers), and compared it honestly against that simple rule, on the same real cities.

the goal was never to reproduce pwtt's own published numbers exactly. they evaluate at 500m grid level with a threshold picked separately per city, we care about individual buildings, that's a different (and for our purpose, more useful) way of measuring it. the point is a fair, consistent comparison between the baseline and our model, both measured the same way: leave one city out cross validation across all 19 cities (every city gets a turn being the "new, unseen place" a model never saw during training).

what was built:
- downloaded and verified the pwtt benchmark data, no account needed for this part
- pulled raw sentinel-1 time series for all 19 cities ourselves, 1.8gb total
- found and fixed two real data quality issues in pwtt's own files (duplicate building ids, and buildings with only one satellite pass having an undefined standard deviation)
- built a feature table from the raw signal, trained 3 models, evaluated with leave one city out cross validation
- logged all results into mlflow

**the honest result:** across all 19 cities, pwtt's simple baseline wins on average (mean AUC 73.8 vs our best model, gradient boosting, at 72.4). our model wins outright on 7 of 19 cities. this matches pwtt's own published finding that a simple statistical method rivals more complex ones here. it's not a failed experiment, it's an independent confirmation of something already published.

**we investigated whether a "use the model above some damage rate threshold, baseline below it" switching rule could turn this into a clean recommendation. it can't, honestly.** searching over a few candidate thresholds on the same 19 cities found a threshold that looked good (a small apparent improvement), but a permutation test (`scripts/permutation_test.py`, randomly shuffle which city gets which damage rate, repeat the same threshold search thousands of times) showed that 26.6% of purely random, meaningless shuffles did as well or better than our real result. that's nowhere near convincing, 19 cities isn't enough data to support a threshold based rule with any confidence. the more honest, weaker claim that does hold up: a trained model's usefulness relative to the baseline swings wildly in cities with very few actual damaged buildings (under about 90), which is a sample size effect, not a real pattern either way. at higher damage rates it's calmer and leans slightly toward the model (4 wins, 3 losses among the 7 cities over 5% damage), suggestive but not proven.

**this part is considered finished.** real ground truth, honest comparison, an honest null result on top (the switching rule), nothing left unresolved.

## part 2: extending to the present day, exploratory, not validated

once the core model was trained on 2022 data, the obvious next question was whether we could point it at fresh, current sentinel-1 data and get a present day damage estimate. there's no unosat assessment of today to check an answer against, so this can only ever be an illustration, not a validated result. testing it seriously surfaced real problems, which is the actual point of keeping this section.

**first check, does it even look plausible.** applied the model to fresh 2026 data for rubizhne, sievierodonetsk and mariupol (all real front line cities). all three came back flagged far more damaged now than their real 2022 assessment found. interesting, but unconfirmable on its own.

**second check, a sanity test against cities that stayed quiet.** ran the same pipeline on kremenchuk and sumy, two cities with only 0.1% real damage in 2022 and no history of being front line combat zones. if the method were reliable, these should stay low. they did not, both came back 45 to 48% flagged, nearly as high as the genuinely war torn cities. that's the signal something is wrong, not a quiet confirmation that distant cities are also destroyed.

**bug 1, found and fixed:** an early mariupol pull used a different satellite orbit than its 2022 baseline, corrupting 95% of its readings. fixed by forcing every fresh pull to reuse the exact same orbit as that city's original baseline.

**bug 2, found and fixed, confirmed with a clean before and after:** a feature measuring how much a building's signal wobbled across satellite passes was built from a different *number* of passes than training ever saw. with very few passes (common in 2022, coverage was sparser then) the spread you can measure is naturally small, almost by mathematical necessity, not because the building was stable. the fresh pull had more passes available (coverage is denser now, likely a new satellite joined the fleet), which reveals a bigger, probably more honest spread the model had simply never seen in training. matching each city's exact 2022 pass count fixed this, not because the new data was wrong, but because it made the comparison fair again. proof it mattered: kremenchuk dropped from 45.2% flagged to **0.8%**, almost exactly matching its true 0.1%. a clean, complete resolution for that city.

**bug 3, found and partially fixed:** the same pass count mismatch existed in a second, longer reference window used by the simpler pwtt style comparison rule. fixing it helped (kremenchuk's pwtt style reading went from 19.3% to 15.8%) but far less dramatically than bug 2 did.

**what did not get fixed, and why we stopped:** sumy stayed elevated (about 30% with our model, about 14% with the pwtt style rule) even after both fixes, despite having the same kind of control city profile as kremenchuk. dug into this and found something that changes the interpretation. sumy was not actually a quiet city after 2022. it suffered a deadly missile strike in november 2024, the deadliest attack on ukrainian civilians since 2023 in april 2025 (35 killed, 20 buildings damaged), and further strikes in june 2025. kremenchuk was also repeatedly hit (an oil refinery targeted by 69 missiles and 260+ drones over the war, residential strikes in 2024 to 2026). so "control city" was the wrong mental model. these places kept getting hit, just not in the sustained urban combat way mariupol or sievierodonetsk were. some of sumy's elevated reading is plausibly real, accumulated damage since 2022. some is plausibly still pipeline noise (building geolocation imprecision, residual seasonal effects, or something not yet found). we checked the most direct way to tell these apart, pulling multiple years of data and looking for discrete jumps lined up with the known strike dates. see the result below. after that check we deliberately stopped chasing this further. an honestly unresolved question is a legitimate way to end a clearly scoped side investigation, which is why this is kept separate from part 1.

**multi year check (2023 to 2026), looking for jumps at the known strike dates:**

pulled summer snapshots for kremenchuk (2023, 2024, 2025, 2026) and windows bracketing sumy's two worst documented strikes (nov 2024, apr 2025), plus a 2023 baseline, and tracked the share of buildings showing an elevated change signal (`combined_z > 2`) in each period. this is the raw signal, not run through the trained model.

| city | period | % of buildings elevated |
|---|---|---|
| kremenchuk | 2023 summer | 9.8% |
| kremenchuk | 2024 summer | 10.0% |
| kremenchuk | 2025 summer | 9.9% |
| kremenchuk | 2026 current | 10.4% |
| sumy | 2023 baseline | 5.2% |
| sumy | around nov 2024 strike | 7.3% |
| sumy | around apr 2025 strike | 6.2% |
| sumy | 2026 current | 8.6% |

**kremenchuk is flat across all 4 years.** no trend, no jump, consistent with its already resolved near zero prediction. **sumy shows a real, gradual increase over time** (5.2% to 8.6%) but not a discrete spike lining up with either strike date specifically. likely reason: the april 2025 strike damaged about 20 buildings out of sumy's 28,265. a single building scale event completely dilutes into a city wide average, so this test was never sensitive enough to isolate it. the gradual rise is more consistent with something diffuse accumulating over years (scattered lower intensity strikes, gradual disrepair, or still uncorrected pipeline drift) than one clean cause.

**conclusion, and where we stopped:** sumy is genuinely changing over time in a way kremenchuk isn't. that's real evidence the elevated reading isn't pure noise. but this doesn't cleanly explain its full size, and per the plan for this investigation, we stopped here rather than keep chasing it. an honestly unresolved question is an acceptable way to end this side investigation. it would not be an acceptable way to end part 1.

**bottom line for this section:** two real bugs found and fixed, one real finding that changed our assumptions (these weren't actually quiet cities), and one open question left honestly unresolved rather than forced to a tidy answer. not something to present as a working present day damage detector.

## repo layout, roughly in the order you'd want to read it

```
scripts/download_benchmark.py         step 1: downloads pwtt's real benchmark data (no account needed)
src/data/benchmark_labels.py          loads a city's csv into a table, computes pwtt's baseline rule
scripts/explore_benchmark.py          step 2: inspect the real data, draw a damage map, sanity check pwtt's own baseline

src/data/sentinel1_gee.py             step 3: talks to google earth engine, pulls raw VV/VH radar readings per building per date
scripts/pull_sentinel1_timeseries.py  runs step 3 for one city (used to test before committing to all 19)
scripts/pull_all_sentinel1.py         runs step 3 for all 19 cities, resumable if interrupted

src/data/build_features.py            step 4: turns the raw per date readings into one row per building (before/after summary stats, this is our own feature engineering)
scripts/build_training_table.py       runs step 4 for all 19 cities, saves data/features.csv

scripts/train_and_evaluate.py         step 5: defines the 3 models and the feature list, trains and tests on a handful of held out cities (early version, superseded by the next script)
scripts/cross_validate_cities.py      step 6: the real evaluation, leave one city out across all 19 cities, saves data/cross_validation_results.csv
scripts/log_to_mlflow.py              step 7: logs the cross validation results into mlflow (doesn't retrain anything, just records what step 6 already found)

scripts/switching_rule.py             step 8: investigates the damage rate switching idea
scripts/permutation_test.py           step 9: checks whether the switching rule's apparent improvement is real or just noise (it's noise)

scripts/plot_city_comparison.py                draws one city as 3 maps: real damage vs pwtt's guess vs our model's guess
scripts/plot_damage_rate_vs_performance.py     chart showing where our model helps or doesn't, across all 19 cities
scripts/build_interactive_map.py               part 2: interactive html map per city, buildings colored by whether a method got them right
scripts/pull_current_snapshot.py               part 2: pulls a fresh, recent sentinel-1 window for one city, forces the same orbit as that city's original baseline
scripts/predict_current_snapshot.py            part 2: runs the trained model and a pwtt style rule on a fresh snapshot, draws the result, no ground truth exists for this
scripts/plot_multi_year_trend.py               part 2: compares the raw signal across several years, used for the sumy/kremenchuk multi year check

data/benchmark/one_month/           pwtt's csvs: building shape, real damage label, pwtt's own precomputed stats (downloaded, not in git)
data/sentinel1/                     our own raw pull: VV/VH per building per date (not in git)
data/features.csv                   our engineered feature table, one row per building (not in git)
data/cross_validation_results.csv   one row per city, all 4 methods' metrics (not in git)
```

## running it

```
pip install -r requirements.txt
python scripts/download_benchmark.py
python scripts/explore_benchmark.py --cities mariupol bucha irpin --plot
```

everything past that needs a free google earth engine account (see the section above on `src/data/sentinel1_gee.py`). run `earthengine authenticate` once, then:

```
python scripts/pull_all_sentinel1.py --project <your-gee-project-id>
python scripts/build_training_table.py
python scripts/cross_validate_cities.py
python scripts/log_to_mlflow.py   # then run: mlflow ui
```
