
# LOAD PACKAGES -----------------------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,
  timetk,
  tsibble,
  tsibbledata,
  fastDummies,
  skimr,
  data.table,
  recipes,
  tidymodels,
  modeltime,
  tictoc,
  future,
  doFuture,
  plotly,
  modeltime.ensemble
)

# LOAD DATA
aus_retail_tbl <- tsibbledata::aus_retail %>%
  timetk::tk_tbl()

# FILTER FOR SPECIFIC STATES
monthly_retail_tbl <- aus_retail_tbl %>%
  filter(State == "Australian Capital Territory") %>%
  mutate(Month = as.Date(Month)) %>%
  mutate(Industry = as_factor(Industry)) %>%
  select(Month, Industry, Turnover)

monthly_retail_tbl

myskim <- skim_with(numeric = sfl(max, min), append = TRUE)

Industries <- unique(monthly_retail_tbl$Industry)

# CREATE FEATURE ENGINEERING TABLE ----------------------------------------

groups <- lapply(X = 1:length(Industries), FUN = function(x) {
  monthly_retail_tbl %>%
    filter(Industry == Industries[x]) %>%
    arrange(Month) %>%
    mutate(Turnover = log1p(x = Turnover)) %>%
    mutate(Turnover = standardize_vec(Turnover)) %>%
    future_frame(Month, .length_out = "12 months", .bind_data = TRUE) %>%
    mutate(Industry = Industries[x]) %>%
    tk_augment_fourier(.date_var = Month, .periods = 12, .K = 1) %>%
    tk_augment_lags(.value = Turnover, .lags = c(12, 13)) %>%
    tk_augment_slidify(
      .value = c(Turnover_lag12, Turnover_lag13),
      .f = ~ mean(.x, na.rm = TRUE),
      .period = c(3, 6, 9, 12),
      .partial = TRUE,
      .align = "center"
    )
})

# IMPUTE MISSING VALUES FOR THE LAGGED AND ROLLING LAG PREDICTORS
groups_fe_tbl <- bind_rows(groups) %>%
  rowid_to_column(var = "rowid") %>%
  group_by(Industry) %>%
  mutate_at(vars(Turnover_lag12:Turnover_lag13_roll_12), .funs = ts_impute_vec, period = 12) %>%
  ungroup()

tmp <- monthly_retail_tbl %>%
  group_by(Industry) %>%
  arrange(Month) %>%
  mutate(Turnover = log1p(x = Turnover)) %>%
  group_map(~ c(
    mean = mean(.x$Turnover, na.rm = TRUE),
    sd = sd(.x$Turnover, na.rm = TRUE)
  )) %>%
  bind_rows()

std_mean <- tmp$mean

std_sd <- tmp$sd

rm("tmp")


# CREATE PREPARED AND FUTURE DATASETS -------------------------------------

# RETAIN THE ROWS WHERE THERE IS NO NA VALUES IN TURNOVER I.E. REMOVE THE FUTURE DATASET THAT WAS ADDED DURING FEATURE ENGINEERING
data_prepared_tbl <- groups_fe_tbl %>%
  filter(!is.na(Turnover)) %>%
  drop_na()

# RETAIN THE ROWS THAT WERE ADDED DURING FEATURE ENGINEERING
future_tbl <- groups_fe_tbl %>%
  filter(is.na(Turnover))


# CREATE THE TRAIN AND TEST DATASETS --------------------------------------

splits <- data_prepared_tbl %>%
  time_series_split(Month,
    assess = "86 months",
    cumulative = TRUE
  )

splits

splits %>%
  tk_time_series_cv_plan() %>%
  glimpse()


# CREATE PREPROCESSING RECIPES --------------------------------------------

recipe_spec <- recipe(Turnover ~ ., data = training(splits)) %>%
  update_role(rowid, new_role = "indicator") %>%
  step_other(Industry) %>%
  step_timeseries_signature(Month) %>%
  step_rm(matches("(.xts$)|(.iso$)|(hour)|(minute)|(second)|(day)|(week)|(am.pm)")) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_normalize(Month_index.num, Month_year)

pre_norm <- recipe(Turnover ~ ., data = training(splits)) %>%
  step_timeseries_signature(Month) %>%
  prep() %>%
  juice() %>%
  myskim()

Month_index.num_limit_lower <- pre_norm %>%
  filter(skim_variable == "Month_index.num") %>%
  select(numeric.min)

Month_index.num_limit_upper <- pre_norm %>%
  filter(skim_variable == "Month_index.num") %>%
  select(numeric.max)

Month_year_limit_lower <- pre_norm %>%
  filter(skim_variable == "Month_year") %>%
  select(numeric.min)

Month_year_limit_upper <- pre_norm %>%
  filter(skim_variable == "Month_year") %>%
  select(numeric.max)

# SAVE FEATURE ENGINEERING ------------------------------------------------


feature_engineering_artifacts_list <- list(
  # DATA
  data = list(
    data_prepared_tbl = data_prepared_tbl,
    future_tbl = future_tbl,
    industries = Industries
  ),
  # RECIPES
  recipes = list(
    recipe_spec = recipe_spec
  ),
  # SPLITS
  splits = splits,
  # INVERSION PARAMETERS
  standardize = list(
    std_mean = std_mean,
    std_sd   = std_sd
  ),
  normalize = list(
    Month_index.num_limit_lower = Month_index.num_limit_lower,
    Month_index.num_limit_upper = Month_index.num_limit_upper,
    Month_year_limit_lower = Month_year_limit_lower,
    Month_year_limit_upper = Month_year_limit_upper
  )
)

feature_engineering_artifacts_list %>%
  write_rds("feature_engineering_artifacts_list.rds")


# LOAD ARTIFACTS ----------------------------------------------------------

artifacts <- read_rds("feature_engineering_artifacts_list.rds")

splits <- artifacts$splits
recipe_spec <- artifacts$recipes$recipe_spec
Industries <- artifacts$data$industries

# CREATE WORKFLOWS --------------------------------------------------------

# RANDOM FOREST WORKFLOW
tic()
wflw_fit_rf <- workflow() %>%
  add_model(
    spec = rand_forest(
      mode = "regression"
    ) %>%
      set_engine("ranger")
  ) %>%
  add_recipe(recipe_spec) %>%
  update_recipe(recipe_spec %>% step_rm(Month)) %>%
  fit(training(splits))
toc()

wflw_fit_rf

# XGBOOST WORKFLOW
tic()
wflw_fit_xgboost <- workflow() %>%
  add_model(
    spec = boost_tree(
      mode = "regression"
    ) %>%
      set_engine("xgboost")
  ) %>%
  add_recipe(recipe_spec) %>%
  update_recipe(recipe_spec %>% step_rm(Month)) %>%
  fit(training(splits))
toc()

wflw_fit_xgboost

# PROPHET WORKFLOW
tic()
wflw_fit_prophet <- workflow() %>%
  add_model(
    spec = prophet_reg(
      seasonality_daily  = FALSE,
      seasonality_weekly = FALSE,
      seasonality_yearly = TRUE
    ) %>%
      set_engine("prophet")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))
toc()

wflw_fit_prophet

# PROPHET BOOST WORKFLOW
tic()
wflw_fit_prophet_boost <- workflow() %>%
  add_model(
    spec = prophet_boost(
      seasonality_daily  = FALSE,
      seasonality_weekly = FALSE,
      seasonality_yearly = FALSE
    ) %>%
      set_engine("prophet_xgboost")
  ) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))
toc()

wflw_fit_prophet_boost


# MODELTIME TABLE ---------------------------------------------------------

submodels_tbl <- modeltime_table(
  wflw_fit_rf,
  wflw_fit_xgboost,
  wflw_fit_prophet,
  wflw_fit_prophet_boost
)

# CALIBRATION TABLE -------------------------------------------------------

calibrated_wflws_tbl <- submodels_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibrated_wflws_tbl

calibrated_wflws_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)


# SAVE WORKFLOW -----------------------------------------------------------

workflow_artifacts <- list(
  workflows = list(
    wflw_random_forest = wflw_fit_rf,
    wflw_xgboost = wflw_fit_xgboost,
    wflw_prophet = wflw_fit_prophet,
    wflw_prophet_boost = wflw_fit_prophet_boost
  ),
  calibration = list(calibration_tbl = calibrated_wflws_tbl)
)

workflow_artifacts %>%
  write_rds("workflow_artifacts_list.rds")


# READ IN WORKFLOW ARTIFACTS ----------------------------------------------

wflw_artifacts <- read_rds("workflow_artifacts_list.rds")

wflw_artifacts$calibration$calibration_tbl %>%
  modeltime_accuracy(testing(splits)) %>%
  arrange(rmse)


# SET UP CROSS VALIDATION PLAN --------------------------------------------

set.seed(123)

resamples_kfold <- training(splits) %>%
  vfold_cv(v = 10)

# resamples_kfold %>%
#   tk_time_series_cv_plan() %>%
#   filter(Industry == Industries[1]) %>%
#   plot_time_series_cv_plan(.date_var = Month,
#                            .value = Turnover,
#                            .facet_ncol = 2)


# PROPHET BOOST PARAMETER TUNING -----------------------------------------

model_spec_prophet_boost_tune <- prophet_boost(
  mode = "regression",
  changepoint_num = tune(),
  seasonality_yearly = FALSE,
  seasonality_weekly = FALSE,
  seasonality_daily = FALSE,
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune()
) %>%
  set_engine("prophet_xgboost")

wflw_spec_prophet_boost_tune <- workflow() %>%
  add_model(model_spec_prophet_boost_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)

wflw_spec_prophet_boost_tune

# artifacts$recipes$recipe_spec %>%
#   update_role(Month, new_role = "indicator") %>%
#   prep() %>%
#   summary() %>%
#   group_by(role) %>%
#   summarise(n=n())

# GRID SPECIFICATION - PROPHET BOOST

# ROUND 1

set.seed(123)
pb_grid_spec_1 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_prophet_boost_tune) %>%
    update(mtry = mtry(range = c(1, 49))),
  size = 10
)

pb_grid_spec_1

registerDoFuture()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_prophet_boost_1 <- wflw_spec_prophet_boost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = pb_grid_spec_1,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

# tune_results_prophet_boost_1 %>%
#   show_best("rmse", n = Inf)
#
# tune_results_prophet_boost_1 %>%
#   show_best("rsq", n = Inf)

pb_gr1 <- tune_results_prophet_boost_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(pb_gr1)


# ROUND 2

set.seed(123)
pb_grid_spec_2 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_prophet_boost_tune) %>%
    update(
      mtry = mtry(range = c(1, 49)),
      learn_rate = learn_rate(range = c(-2.0, -1.0))
    ),
  size = 10
)

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_prophet_boost_2 <- wflw_spec_prophet_boost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = pb_grid_spec_2,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

tune_results_prophet_boost_2 %>%
  show_best("rsq", n = 2)

tune_results_prophet_boost_2 %>%
  show_best("rmse", n = 2)

pb_gr2 <- tune_results_prophet_boost_2 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(pb_gr2)

# ROUND 3 - FIXING TREE PARAMETER
set.seed(123)
pb_grid_spec_3 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_prophet_boost_tune) %>%
    update(
      mtry = mtry(range = c(1, 49)),
      learn_rate = learn_rate(range = c(-2.0, -1.0)),
      trees = trees(range = c(1500, 1770))
    ),
  size = 10
)

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_prophet_boost_3 <- wflw_spec_prophet_boost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = pb_grid_spec_3,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

# tune_results_prophet_boost_3 %>%
#   show_best("rmse", n = 2)
#
# tune_results_prophet_boost_3 %>%
#   show_best("rsq", n = 2)

# SELECT THE BEST PROPHET BOOST MODEL
set.seed(123)
wflw_fit_prophet_boost_tuned <- wflw_spec_prophet_boost_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_boost_3, "rmse", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_boost_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# FIT THE ROUND 3 BEST PROPHET BOOST RSQ MODEL
set.seed(123)
wflw_fit_prophet_boost_tuned_rsq <- wflw_spec_prophet_boost_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_boost_3, "rsq", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_boost_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# SAVE PROPHET BOOST TUNING ARTIFACTS
tuned_prophet_xgboost <- list(
  # WORKFLOW SPEC
  tune_wkflw_spec = wflw_spec_prophet_boost_tune,
  # GRID SPEC
  tune_grid_spec = list(
    round1 = pb_grid_spec_1,
    round2 = pb_grid_spec_2,
    round3 = pb_grid_spec_3
  ),
  # TUNING RESULTS
  tune_results = list(
    round1 = tune_results_prophet_boost_1,
    round2 = tune_results_prophet_boost_2,
    round3 = tune_results_prophet_boost_3
  ),
  # TUNED WORKFLOW FIT
  tune_wflw_fit = wflw_fit_prophet_boost_tuned,
  # FROM FEATURE ENGINEERING
  splits = artifacts$splits,
  data = artifacts$data,
  recipes = artifacts$recipes,
  standardize = artifacts$standardize,
  normalize = artifacts$normalize
)

tuned_prophet_xgboost %>%
  write_rds("tuned_prophet_xgboost.rds")


# RANDOM FOREST PARAMETER TUNING -----------------------------------------
# ROUND 1
model_spec_random_forest_tune <- parsnip::rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = 1000,
  min_n = tune()
) %>%
  set_engine("ranger")

wflw_spec_random_forest_tune <- workflow() %>%
  add_model(model_spec_random_forest_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)

wflw_spec_random_forest_tune

extract_parameter_set_dials(model_spec_random_forest_tune)

artifacts$recipes$recipe_spec %>%
  update_role(Month, new_role = "indicator") %>%
  prep() %>%
  summary() %>%
  group_by(role) %>%
  summarise(n = n())

set.seed(123)
rf_grid_spec_1 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_random_forest_tune) %>%
    update(mtry = mtry(range = c(1, 49))),
  size = 10
)

rf_grid_spec_1

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_random_forest_1 <- wflw_spec_random_forest_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = rf_grid_spec_1,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

tune_results_random_forest_1 %>%
  show_best("rmse", n = Inf)

tune_results_random_forest_1 %>%
  show_best("rsq", n = Inf)

rf_gr1 <- tune_results_random_forest_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(rf_gr1)

# ROUND 2

set.seed(123)
rf_grid_spec_2 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_random_forest_tune) %>%
    update(mtry = mtry(range = c(17, 28))),
  size = 10
)

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_random_forest_2 <- wflw_spec_random_forest_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = rf_grid_spec_2,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

tune_results_random_forest_2 %>%
  show_best("rmse", n = Inf)

tune_results_random_forest_2 %>%
  show_best("rsq", n = Inf)

rf_gr2 <- tune_results_random_forest_2 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(rf_gr2)

# FITTING ROUND 2 BEST RMSE MODEL
set.seed(123)
wflw_fit_random_forest_tuned <- wflw_spec_random_forest_tune %>%
  finalize_workflow(
    select_best(tune_results_random_forest_2, "rmse", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_random_forest_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# FITTING ROUND 2 BEST RSQ MODEL
set.seed(123)
wflw_fit_random_forest_tuned_rsq <- wflw_spec_random_forest_tune %>%
  finalize_workflow(
    select_best(tune_results_random_forest_2, "rsq", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_random_forest_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

tuned_random_forest <- list(
  # WORKFLOW SPEC
  tune_wkflw_spec = wflw_spec_random_forest_tune,
  # GRIC SPEC
  tune_grid_spec = list(
    round1 = rf_grid_spec_1,
    round2 = rf_grid_spec_2
  ),
  # TUNING RESULTS
  tune_results = list(
    round1 = tune_results_random_forest_1,
    round2 = tune_results_random_forest_2
  ),
  # TUNED WORKFLOW FIT
  tune_wflw_fit = wflw_fit_random_forest_tuned,
  # FROM FEATURE ENGINEERING
  splits = artifacts$splits,
  data = artifacts$data,
  recipes = artifacts$recipes,
  standardize = artifacts$standardize,
  normalize = artifacts$normalize
)

tuned_random_forest %>%
  write_rds("tuned_random_forest.rds")


# PROPHET PARAMETER TUNING -----------------------------------------------

model_spec_prophet_tune <- prophet_reg(
  mode = "regression",
  growth = "linear",
  changepoint_num = tune(),
  changepoint_range = tune(),
  seasonality_yearly = TRUE,
  seasonality_weekly = FALSE,
  seasonality_daily = FALSE
) %>%
  set_engine("prophet")

wflw_spec_prophet_tune <- workflow() %>%
  add_model(model_spec_prophet_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec)

wflw_spec_prophet_tune

# ROUND 1
set.seed(123)
prophet_grid_spec_1 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_prophet_tune) %>%
    update(
      changepoint_num = changepoint_num(range = c(0L, 50L), trans = NULL),
      changepoint_range = changepoint_range(range = c(0.7, 0.9), trans = NULL)
    ),
  size = 10
)

prophet_grid_spec_1

registerDoFuture()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_prophet_1 <- wflw_spec_prophet_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = prophet_grid_spec_1,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

tune_results_prophet_1 %>%
  show_best("rmse", n = Inf)

tune_results_prophet_1 %>%
  show_best("rsq", n = Inf)

prophet_gr1 <- tune_results_prophet_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(prophet_gr1)

# FITTING ROUND 1 BEST RMSE MODEL
set.seed(123)
wflw_fit_prophet_tuned <- wflw_spec_prophet_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_1, "rmse", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# FITTING ROUND 1 BEST RSQ MODEL
set.seed(123)
wflw_fit_prophet_tuned_rsq <- wflw_spec_prophet_tune %>%
  finalize_workflow(
    select_best(tune_results_prophet_1, "rsq", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_prophet_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

tuned_prophet <- list(
  # WORKFLOW SPEC
  tune_wkflw_spec = wflw_spec_prophet_tune,
  # GRIC SPEC
  tune_grid_spec = list(
    round1 = prophet_grid_spec_1
  ),
  # TUNING RESULTS
  tune_results = list(
    round1 = tune_results_prophet_1
  ),
  # TUNED WORKFLOW FIT
  tune_wflw_fit = wflw_fit_prophet_tuned,
  # FROM FEATURE ENGINEERING
  splits = artifacts$splits,
  data = artifacts$data,
  recipes = artifacts$recipes,
  standardize = artifacts$standardize,
  normalize = artifacts$normalize
)

tuned_prophet %>%
  write_rds("tuned_prophet.rds")


# XGBOOST PARAMETER TUNING ------------------------------------------------

model_spec_xgboost_tune <- boost_tree(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost")

model_spec_xgboost_tune

wflw_spec_xgboost_tune <- workflow() %>%
  add_model(model_spec_xgboost_tune) %>%
  add_recipe(artifacts$recipes$recipe_spec) %>%
  update_recipe(artifacts$recipes$recipe_spec %>% step_rm(Month))

artifacts$recipes$recipe_spec %>%
  step_rm(Month) %>%
  prep() %>%
  summary() %>%
  group_by(role) %>%
  summarise(n = n())

extract_parameter_set_dials(model_spec_xgboost_tune)

# ROUND 1

set.seed(123)
xgboost_grid_spec_1 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_xgboost_tune) %>%
    update(mtry = mtry(range = c(1, 49))),
  size = 10
)

xgboost_grid_spec_1

registerDoFuture()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

extract_preprocessor(wflw_spec_xgboost_tune)

tic()
tune_results_xgboost_1 <- wflw_spec_xgboost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = xgboost_grid_spec_1,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

# tune_results_xgboost_1 %>%
#   show_best("rmse", n = Inf)
#
# tune_results_xgboost_1 %>%
#   show_best("rsq", n = Inf)

xgboost_gr1 <- tune_results_xgboost_1 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(xgboost_gr1)

# ROUND 2

set.seed(123)
xgboost_grid_spec_2 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_xgboost_tune) %>%
    update(
      mtry = mtry(range = c(1, 49)),
      learn_rate = learn_rate(range = c(-1.5, -0.5))
    ),
  size = 10
)

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_xgboost_2 <- wflw_spec_xgboost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = xgboost_grid_spec_2,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

tune_results_xgboost_2 %>%
  show_best("rsq", n = 2)

tune_results_xgboost_2 %>%
  show_best("rmse", n = 2)

xgboost_gr2 <- tune_results_xgboost_2 %>%
  autoplot() +
  geom_smooth(se = FALSE)

ggplotly(xgboost_gr2)

# ROUND 3 - FIXING TREE PARAMETER
set.seed(123)
xgboost_grid_spec_3 <- grid_latin_hypercube(
  extract_parameter_set_dials(model_spec_xgboost_tune) %>%
    update(
      mtry = mtry(range = c(1, 49)),
      learn_rate = learn_rate(range = c(-1.5, -0.5)),
      trees = trees(range = c(1000, 1500))
    ),
  size = 10
)

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(parallel::detectCores())
)

tic()
tune_results_xgboost_3 <- wflw_spec_xgboost_tune %>%
  tune_grid(
    resamples = resamples_kfold,
    grid = xgboost_grid_spec_3,
    control = control_grid(
      verbose = TRUE,
      allow_par = TRUE
    )
  )
toc()

plan(strategy = sequential)

# tune_results_prophet_boost_3 %>%
#   show_best("rmse", n = 2)
#
# tune_results_prophet_boost_3 %>%
#   show_best("rsq", n = 2)

# SELECT THE BEST PROPHET BOOST MODEL
set.seed(123)
wflw_fit_xgboost_tuned <- wflw_spec_xgboost_tune %>%
  finalize_workflow(
    select_best(tune_results_xgboost_3, "rmse", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_xgboost_tuned) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# FIT THE ROUND 3 BEST PROPHET BOOST RSQ MODEL
set.seed(123)
wflw_fit_xgboost_tuned_rsq <- wflw_spec_xgboost_tune %>%
  finalize_workflow(
    select_best(tune_results_xgboost_3, "rsq", n = 1)
  ) %>%
  fit(training(splits))

modeltime_table(wflw_fit_xgboost_tuned_rsq) %>%
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy()

# SAVE PROPHET BOOST TUNING ARTIFACTS
tuned_xgboost <- list(
  # WORKFLOW SPEC
  tune_wkflw_spec = wflw_spec_xgboost_tune,
  # GRID SPEC
  tune_grid_spec = list(
    round1 = xgboost_grid_spec_1,
    round2 = xgboost_grid_spec_2,
    round3 = xgboost_grid_spec_3
  ),
  # TUNING RESULTS
  tune_results = list(
    round1 = tune_results_xgboost_1,
    round2 = tune_results_xgboost_2,
    round3 = tune_results_xgboost_3
  ),
  # TUNED WORKFLOW FIT
  tune_wflw_fit = wflw_fit_xgboost_tuned,
  # FROM FEATURE ENGINEERING
  splits = artifacts$splits,
  data = artifacts$data,
  recipes = artifacts$recipes,
  standardize = artifacts$standardize,
  normalize = artifacts$normalize
)

tuned_xgboost %>%
  write_rds("tuned_xgboost.rds")


# MODELTIME & CALIBRATION TABLES ------------------------------------------

submodels_tbl <- modeltime_table(
  wflw_artifacts$workflows$wflw_random_forest,
  wflw_artifacts$workflows$wflw_xgboost,
  wflw_artifacts$workflows$wflw_prophet,
  wflw_artifacts$workflows$wflw_prophet_boost
)

submodels_all_tbl <- modeltime_table(
  tuned_random_forest$tune_wflw_fit,
  tuned_xgboost$tune_wflw_fit,
  tuned_prophet$tune_wflw_fit,
  tuned_prophet_xgboost$tune_wflw_fit
) %>%
  update_model_description(1, "RANGER - Tuned") %>%
  update_model_description(2, "XGBOOST - Tuned") %>%
  update_model_description(3, "PROPHET W/ REGRESSORS - Tuned") %>%
  update_model_description(4, "PROPHET W/ XGBOOST ERRORS - Tuned") %>%
  combine_modeltime_tables(submodels_tbl)

submodels_all_tbl


# MODEL EVALUATION --------------------------------------------------------

calibration_all_tbl <- submodels_all_tbl %>%
  modeltime_calibrate(testing(splits))

calibration_all_tbl %>%
  modeltime_accuracy() %>%
  arrange(rmse)

calibration_all_tbl %>%
  modeltime_accuracy() %>%
  arrange(desc(rsq))

calibration_all_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy()


# FORECAST PLOTS ----------------------------------------------------------

calibration_all_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = artifacts$data$data_prepared_tbl,
    keep_data   = TRUE
  ) %>%
  filter(Industry == Industries[1]) %>%
  plot_modeltime_forecast(
    .facet_ncol = 4,
    .conf_interval_show = FALSE,
    .interactive = TRUE,
    .title = Industries[1]
  )


# SAVE WORK ---------------------------------------------------------------

workflow_all_artifacts <- list(
  workflows = submodels_all_tbl,
  calibration = calibration_all_tbl
)

workflow_all_artifacts %>%
  write_rds("workflows_NonandTuned_artifacts_list.rds")


# # LOAD CALIBRATION TABLES -------------------------------------------------
# 
# calibration_tbl <- read_rds("workflows_NonandTuned_artifacts_list.rds")
# 
# calibration_tbl <- calibration_tbl$calibration
# 
# calibration_tbl %>%
#   modeltime_accuracy() %>%
#   arrange(rmse)
# 
# 
# # AVERAGE ENSEMBLES -------------------------------------------------------
# 
# ensemble_fit_mean <- calibration_all_tbl %>%
#   ensemble_average(type = "mean")
# 
# ensemble_fit_mean
# 
# ensemble_fit_median <- calibration_all_tbl %>%
#   ensemble_average(type = "median")
# 
# ensemble_fit_median
# 
# # WEIGHTED ENSEMBLE -------------------------------------------------------
# 
# calibration_tbl %>%
#   modeltime_accuracy() %>%
#   mutate(rank = min_rank(-rmse))
# 
# loadings_tbl <- calibration_tbl %>%
#   modeltime_accuracy() %>%
#   mutate(rank = min_rank(-rmse)) %>%
#   select(.model_id, rank)
# 
# ensemble_fit_wt <- calibration_tbl %>%
#   ensemble_weighted(loadings = loadings_tbl$rank)
# 
# ensemble_fit_wt$fit$loadings_tbl
# 
# 
# # MODEL EVALUATION --------------------------------------------------------
# 
# modeltime_table(
#   ensemble_fit_mean,
#   ensemble_fit_median,
#   ensemble_fit_wt
# ) %>%
#   modeltime_calibrate(testing(splits)) %>%
#   modeltime_accuracy(testing(splits)) %>%
#   arrange(rmse)
# 
# calibration_all_tbl <- modeltime_table(
#   ensemble_fit_mean,
#   ensemble_fit_median,
#   ensemble_fit_wt
# ) %>%
#   modeltime_calibrate(testing(splits)) %>%
#   combine_modeltime_tables(calibration_tbl)
# 
# calibration_all_tbl
# 
# calibration_all_tbl %>%
#   modeltime_accuracy(testing(splits)) %>%
#   arrange(rmse)
# 
# calibration_all_tbl %>%
#   modeltime_forecast(
#     new_data    = testing(splits),
#     actual_data = artifacts$data$data_prepared_tbl,
#     keep_data   = TRUE
#   ) %>%
#   filter(
#     str_detect(.model_desc, "WEIGHTED|ACTUAL"),
#     .index >= "2010-01-01"
#   ) %>%
#   group_by(Industry) %>%
#   plot_modeltime_forecast(
#     .facet_ncol = 4,
#     .conf_interval_show = FALSE,
#     .interactive = TRUE,
#     .title = "Average Ensemble Forecast"
#   )


# STACKED ENSEMBLE --------------------------------------------------------

# Load all calibration tables (tuned & non-tuned models)
calibration_tbl <- read_rds("workflows_NonandTuned_artifacts_list.rds")

calibration_tbl <- calibration_tbl$calibration

calibration_tbl %>%
  modeltime_accuracy() %>%
  arrange(rmse)

set.seed(123)
resamples_kfold <- training(splits) %>%
  drop_na() %>%
  vfold_cv(v = 10)

tic()
submodels_resamples_kfold_tbl <- calibration_tbl %>%
  modeltime_fit_resamples(
    resamples = resamples_kfold,
    control   = control_resamples(
      verbose    = TRUE, 
      allow_par  = TRUE,
    )
  )
toc()

# Parallel Processing ----
registerDoFuture()
n_cores <- parallel::detectCores()

plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

tic()
set.seed(123)
ensemble_fit_ranger_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = rand_forest(
      mode = "regression",
      trees = tune(),
      min_n = tune()
    ) %>%
      set_engine("ranger"),
    kfolds  = 10, 
    grid    = 20,
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )
toc()

modeltime_table(
  ensemble_fit_ranger_kfold
) %>%
  modeltime_accuracy(testing(splits))

tic()
set.seed(123)
ensemble_fit_xgboost_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = boost_tree(
      mode = "regression",
      trees          = tune(),
      tree_depth     = tune(),
      learn_rate     = tune(),
      loss_reduction = tune()
    ) %>%
      set_engine("xgboost"),
    kfolds = 10, 
    grid   = 20, 
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )
toc()

modeltime_table(
  ensemble_fit_xgboost_kfold
) %>%
  modeltime_accuracy(testing(splits))

tic()
set.seed(123)
ensemble_fit_svm_kfold <- submodels_resamples_kfold_tbl %>%
  ensemble_model_spec(
    model_spec = svm_rbf(
      mode      = "regression",
      cost      = tune(),
      rbf_sigma = tune(),  
      margin    = tune()
    ) %>%
      set_engine("kernlab"),
    kfold = 10, 
    grid  = 20, 
    control = control_grid(verbose = TRUE, 
                           allow_par = TRUE)
  )
toc()

modeltime_table(
  ensemble_fit_svm_kfold
) %>%
  modeltime_accuracy(testing(splits))

modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
) %>%
  modeltime_accuracy(testing(splits)) %>% 
  arrange(rmse)

loadings_tbl <- modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
) %>% 
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy() %>%
  mutate(rank = min_rank(-rmse)) %>%
  select(.model_id, rank)

stacking_fit_wt <- modeltime_table(
  ensemble_fit_ranger_kfold, 
  ensemble_fit_xgboost_kfold,
  ensemble_fit_svm_kfold
) %>%
  ensemble_weighted(loadings = loadings_tbl$rank)

stacking_fit_wt  %>% 
  modeltime_calibrate(testing(splits)) %>%
  modeltime_accuracy() %>%
  arrange(rmse)

calibration_stacking <- stacking_fit_wt %>% 
  modeltime_table() %>%
  modeltime_calibrate(testing(splits))

calibration_stacking %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = artifacts$data$data_prepared_tbl,
    keep_data   = TRUE 
  ) %>%
  group_by(Industry) %>%
  plot_modeltime_forecast(
    .facet_ncol         = 4, 
    .conf_interval_show = FALSE,
    .interactive        = TRUE,
    .title = "Forecast stacking level model (test data)"
  )


# NEXT 12 MONTH FORECAST --------------------------------------------------

# Toggle ON parallel processing
plan(
  strategy = cluster,
  workers  = parallel::makeCluster(n_cores)
)

# Refit the model on prepared dataset
tic()
set.seed(123)
refit_stacking_tbl <- calibration_stacking %>% 
  modeltime_refit(
    data = artifacts$data$data_prepared_tbl,
    resamples = artifacts$data$data_prepared_tbl %>%
      drop_na() %>%
      vfold_cv(v = 10)
  )
toc()

forecast_stacking_tbl <- refit_stacking_tbl %>%
  modeltime_forecast(
    new_data    = artifacts$data$future_tbl,
    actual_data = artifacts$data$data_prepared_tbl %>%
      drop_na(), 
    keep_data = TRUE
  )

plan(sequential)

lforecasts <- lapply(X = 1:length(Industries), FUN = function(x){
  forecast_stacking_tbl %>%
    filter(Industry == Industries[x]) %>%
    #group_by(Industry) %>%
    mutate(across(.value:.conf_hi,
                  .fns = ~standardize_inv_vec(x = .,
                                              mean = artifacts$standardize$std_mean[x],
                                              sd = artifacts$standardize$std_sd[x]))) %>%
    mutate(across(.value:.conf_hi,
                  .fns = ~expm1(x = .)))
})

forecast_stacking_tbl <- bind_rows(lforecasts)

forecast_stacking_tbl %>%
  group_by(Industry) %>%
  plot_modeltime_forecast(.title = "Turnover 1-year forecast",     
                          .facet_ncol         = 4, 
                          .conf_interval_show = FALSE,
                          .interactive        = TRUE)

