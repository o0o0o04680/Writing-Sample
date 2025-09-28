import numpy as np, pandas as pd, statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from scipy import stats
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Python\Final_Analysis_Data_2.csv")

## Data Cleaning
data.info()
data = data.drop(columns=['roe','joined_co','page'])

desc = data.describe(include="all")
data = data.replace([np.inf, -np.inf], np.nan)
desc = data.describe(include="all")
print(desc.head())

print(data.isna().sum())
print(data.isna().any().any())    #anyNA()

## According to the ceo  name to fill the N/A
by_ceo = data.groupby("exec_fullname")
cols_to_fill = ['roa','tobin_q','firm_size','leverage','capex_ratio','cash_ratio','op_margin','rd_intensity']
for col in cols_to_fill:
    data[col] = by_ceo[col].transform( lambda x: x.fillna(x.median()))
for col in cols_to_fill:
    data[col] = data[col].fillna(0)
data = data.dropna(subset = ["tdc1", "tdc2", "tdc1_pct"])
print(data.isna().sum())

data = data.drop_duplicates(subset=['gvkey','year'], keep='first')
data = data[data['year'] >= 2004]
for y in [2008, 2020, 2021]:
    data[f"year_{y}"] = (data["year"] ==y).astype(int)
data = data.dropna()
pdata = data.set_index(['gvkey','year']).sort_index()


## Correlation between factors
cor_mat = pdata[["tobin_q","salary","firm_size","leverage","rd_intensity","cash_ratio"]].corr()
cor_mat = cor_mat.round(3)
print(cor_mat)

##VIF

print(sm.__version__)
pdata.loc[:, "lag_salary"] = pdata.groupby(level = 0)["salary"].shift(1)
x = pdata[["lag_salary","firm_size","leverage","rd_intensity","cash_ratio"]]
x= x.replace([np.inf, -np.inf],np.nan).dropna()
x_constant = add_constant(x)
vif_df = pd.DataFrame()
vif_df["Variable"] = x_constant.columns

vif_df["VIF"] = [variance_inflation_factor(x_constant.values, i)
                 for i in range(x_constant.shape[1])]
print(vif_df) 


## Define

def ensure_lags(pdata, cols):
    pdata = pdata.copy()
    for c in cols:
        lagc = f"lag_{c}"
        if lagc not in pdata.columns:
            pdata[lagc] = pdata.groupby(level = 0)[c].shift(1)
    return pdata 


def lag_year_dummies(pdata, y_col = "tobin_q", use_year_dummies = False):
    lag_vars = [f"lag_{c}" for c in lag_base]
    xcols = lag_vars.copy()
    if use_year_dummies:
        xcols += ["year_2008", "year_2020", "year_2021"]
    need = xcols + [y_col]
    pdata_clean =  pdata.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    y = pdata_clean[y_col]
    x = pdata_clean[xcols]
    return pdata_clean, y, x

def fit_fe_model(y, x):
    model = PanelOLS(y, x, entity_effects= True)
    res_dk = model.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2 )
    res_cluster = model.fit(cov_type = "clustered", cluster_entity = True)
    return res_dk, res_cluster

def hausman_test(res_A, res_B, labelA = "A", labelB = "B"):
    bA = res_A.params.copy()
    bB = res_B.params.copy()

    common = bA.index.intersection(bB.index)
    if len(common) == 0:
        raise ValueError("The two models have no common slope parameters to compare.")
    bA = bA[common]
    bB = bB[common]
    VA = res_A.cov.copy().loc[common, common]
    VB = res_B.cov.copy().loc[common, common]

    diff = bA -bB
    Vdiff = VA - VB
    
    try:
        Vinv = np.linalg.inv(Vdiff.values)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(Vdiff.values)

    stat = float(diff.values.T @ Vinv @ diff.values)
    if stat < 0:
        stat = 0.0
    df = len(common)
    pval = float(1.0 - stats.chi2.cdf(stat, df))

    print("Hausman Test")
    print(f"data:  {labelA} and {labelB}")
    print(f"chisq = {stat:.4f}, df = {df}, p-value = {pval:.4g}")
    print("alternative hypothesis: one model is inconsistent")

    return{
        "stat": stat,
        "df": df,
        "pval": pval,
        "compared": f"{labelA} vs {labelB}",
        "params_compared":list(common),
    }    

pdata = data.set_index(['gvkey','year']).sort_index()
lag_base = ['salary','firm_size','leverage','rd_intensity','cash_ratio','ceo_duality']
pdata = ensure_lags(pdata, lag_base)


"""
Lag_independent variables
"""
## Fixed Effect(within)

pdata_clean = pdata.dropna(subset = [f"lag_{col}" for col in ["salary", "firm_size", "leverage", "rd_intensity", "cash_ratio", "ceo_duality"]])

y = pdata_clean["tobin_q"]
x = pdata_clean[["lag_salary", "lag_firm_size", "lag_leverage", "lag_rd_intensity", "lag_cash_ratio", "lag_ceo_duality"]]
model = PanelOLS(y, x, entity_effects = True)

##Discoll-Kraay Robust SE
res_dk = model.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)
print(res_dk.summary)

##Cluster-robust SE
res_cluster = model.fit(cov_type= "clustered", cluster_entity = True)
print(res_cluster.summary)

## Pooled OLS For BG/BP Test
y = pdata_clean["tobin_q"]
x = pdata_clean[["lag_salary", "lag_firm_size", "lag_leverage", "lag_rd_intensity", "lag_cash_ratio", "lag_ceo_duality"]]
x = sm.add_constant(x)
ols_reg = sm.OLS(y, x).fit()

# BG Test (non-panel-aware)
print(acorr_breusch_godfrey(ols_reg, nlags=2)) 

# BP Test
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_reg.resid, ols_reg.model.exog)
print(f"BP: LM={bp_lm:.3f} p={bp_lm_p:.4g}  F={bp_f:.3f} p={bp_f_p:.4g}")

#for bgtest
e = res_dk.resids.rename("e").to_frame()
e["e_lag"] = e.groupby( level = 0)["e"].shift(1)
e = e.dropna()
ar1_fe = PanelOLS(e["e"], e[["e_lag"]], entity_effects = True).fit(cov_type = "clustered", cluster_entity = True)
print(ar1_fe.summary)



### Lag  Salary _ Fixed Effect + salary + Year dummy ( 2008, 2020, 2021)



#FE
pdata_clean, y_y, x_y = lag_year_dummies(pdata, use_year_dummies = True)
res_dk_y, res_cluster_y = fit_fe_model(y_y, x_y)
print(res_dk_y.summary)
print(res_cluster_y.summary)

y_col = "tobin_q"
x_cols = ["lag_salary","lag_firm_size","lag_leverage","lag_rd_intensity","lag_cash_ratio","lag_ceo_duality"]
need = [y_col] + x_cols
pdata_3yr = pdata.dropna(subset = need).copy()
y = pdata_3yr[y_col]
x_base = pdata_3yr[x_cols]

for yy in [2008, 2020, 2021]:
    col = f"year_{yy}"
    if col not in pdata_3yr.columns:
        pdata_3yr[col] = (pdata_3yr.index.get_level_values("year") == yy).astype(int)

x_3yr = pd.concat([x_base, pdata_3yr[["year_2008", "year_2020", "year_2021"]]], axis = 1)
fe_oneway_3yr = PanelOLS(y, x_3yr, entity_effects=True).fit(cov_type= "clustered", cluster_entity = True)
print("\n[FE(one-way) + year(2008,2020,2021) | homoskedastic]")
print(fe_oneway_3yr.summary)

#BP/BG Test
xy_ols = sm.add_constant(x_y)
ols_y = sm.OLS(y_y, xy_ols).fit()

print(acorr_breusch_godfrey(ols_y, nlags = 2))
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_y.resid, ols_y.model.exog)
print(f"[Year dummies] BP: LM = {bp_lm:.3f} p = {bp_lm_p:.4g} | F = {bp_f:.3f} p = {bp_f_p:.4g}")
e_y = res_dk_y.resids.rename("e").to_frame()
e_y["e_lag"] = e_y.groupby(level = 0)["e"].shift(1)
e_y = e_y.dropna()
ar1_fe_y = PanelOLS(e_y["e"], e_y[["e_lag"]], entity_effects = True).fit(cov_type = "clustered", cluster_entity = True)
print(ar1_fe_y.summary)

#### Twoway Effect 
y_col = "tobin_q"
x_cols_lag = ["lag_salary","lag_firm_size","lag_leverage","lag_rd_intensity","lag_cash_ratio","lag_ceo_duality"]
need = [y_col] + x_cols_lag
pdata_tw = pdata.dropna(subset=need).copy()
y_tw = pdata_tw[y_col]
x_tw = pdata_tw[x_cols_lag]
mod_twoway = PanelOLS(y_tw, x_tw, entity_effects = True, time_effects = True)

res_twoway_cl_entity = mod_twoway.fit(cov_type = "clustered", cluster_entity = True)
print("\n[Two-way FE | Clustered by entity]")
print(res_twoway_cl_entity.summary)

res_twoway_2way = mod_twoway.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
print("\n[Two-way FE | Two-way clustered SEs (entity + time)]")
print(res_twoway_2way.summary)

x_ols = sm.add_constant(x_tw, has_constant='add')
ols_pool = sm.OLS(y_tw, x_ols).fit()
print("\n[BG test via pooled OLS (非 panel-aware)]")
print(acorr_breusch_godfrey(ols_pool, nlags=2)) 

bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_pool.resid, ols_pool.model.exog)
print(f"[BP test via pooled OLS] LM={bp_lm:.3f} p={bp_lm_p:.4g} | F={bp_f:.3f} p={bp_f_p:.4g}")

res_twoway_dk = mod_twoway.fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
e_tw = res_twoway_dk.resids.rename("e").to_frame()
e_tw["e_lag"] = e_tw.groupby(level=0)["e"].shift(1)
e_tw = e_tw.dropna()
ar1_fe_tw = PanelOLS(e_tw["e"], e_tw[["e_lag"]], entity_effects=True).fit(
    cov_type="clustered", cluster_entity=True
)
print("\n[AR(1) on FE residuals (two-way FE residual)]")

print(ar1_fe_tw.summary)

##Random Effect

y_col = "tobin_q"
x_cols_lag = ["lag_salary","lag_firm_size","lag_leverage","lag_rd_intensity","lag_cash_ratio","lag_ceo_duality"]
need = [y_col] + x_cols_lag
pdata_re = pdata.dropna(subset=need).copy()
y_re = pdata_re[y_col]
x_re = pdata_re[x_cols_lag]

mod_re = RandomEffects(y_re, x_re)
res_re = mod_re.fit()
print("\n[Random Effects | Default]")
print(res_re.summary)

##BG\BP Test
x_ols_re = sm.add_constant(x_re, has_constant = "add")
ols_pool_re = sm.OLS(y_re, x_ols_re).fit()
print("\n[BG test via pooled OLS (non-panel-aware)]")
print(acorr_breusch_godfrey(ols_pool_re, nlags = 2))
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_pool_re.resid, ols_pool_re.model.exog)
print(f"[BP test via pooled OLS] LM = {bp_lm:.3f} p = {bp_lm_p:.4g} | F = {bp_f:.3f} p = {bp_f_p:.4g}")

res_re_cl = mod_re.fit(cov_type="clustered", cluster_entity = True)
print("\n[Random Effects | Clustered by entity]")
print(res_re_cl.summary)

res_re_dk = mod_re.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)
print("\n[Random Effects | Driscoll-Kraay]")
print(res_re_dk.summary)

##Husaman 

y_col = "tobin_q"
x_cols = ["lag_salary", "lag_firm_size", "lag_leverage", "lag_rd_intensity", "lag_cash_ratio", "lag_ceo_duality"]
need = [y_col] + x_cols
pdata_h = pdata.dropna(subset=need).copy()
y = pdata_h[y_col]
x = pdata_h[x_cols]
fe_oneway = PanelOLS(y, x, entity_effects = True).fit()
fe_twoway = PanelOLS(y, x, entity_effects = True, time_effects = True).fit()

years = pdata_h.index.get_level_values("year")
dummies_full = pd.get_dummies(
    x.index.get_level_values("year"),
    prefix = "yr",
    drop_first=True
)
dummies_full.index = x.index 
x_oneyear = x.join(dummies_full)
re_mod = RandomEffects(y, x).fit()




haus1 = hausman_test(fe_oneway, re_mod, "FE(one-way)", "RE")
haus2 = hausman_test(fe_oneway_3yr, fe_oneway, "FE(one-way + year)", "FE(one-way)")
haus3 = hausman_test(fe_twoway, fe_oneway, "FE(two-way)", "FE(one-way)")
haus4 = hausman_test(fe_twoway, fe_oneway_3yr, "FE(two-way)", "FE(one-way + year)")

for i, h in enumerate((haus1, haus2, haus3, haus4), start=1):
    print(f"Hausman {i} — {h['compared']}: chi2 = {h['stat']:.3f}, df = {h['df']}, p = {h['pval']:.4g}")
    print("  params compared:", h['params_compared'])



'''
 log difference
'''

df = data.sort_values(["gvkey", "year"]).copy() ## Ensure that the data is a data
g = df.groupby("gvkey", sort = False)

def logdiff_pos(series, lag_series):
    return np.where((series > 0) &(lag_series > 0),
                    np.log(series) - np.log(lag_series),
                    np.nan)

df["dl_salary"] = logdiff_pos(df["salary"], g["salary"].shift(1))
df["dl_firm_size"] = logdiff_pos(df["firm_size"], g["firm_size"].shift(1))
df["dl_leverage"] = logdiff_pos(df["leverage"], g["leverage"].shift(1))
df["dl_rd_intensity"] = logdiff_pos(df["rd_intensity"], g["rd_intensity"].shift(1))
df["dl_cash_ratio"] = logdiff_pos(df["cash_ratio"], g["cash_ratio"].shift(1))
df["dl_ceo_duality"] = df["ceo_duality"]
df["dl_tobin"] = logdiff_pos(df["tobin_q"], g["tobin_q"].shift(1))

##two-way FE
keep_cols = ["dl_tobin", "dl_salary", "dl_firm_size", "dl_leverage", "dl_rd_intensity", "dl_cash_ratio"]
mask = np.isfinite(df[keep_cols]).all(axis = 1) & df["dl_ceo_duality"].notna()
data_clean = df.loc[mask].copy()
pdata_ld = data_clean.set_index(["gvkey", "year"]).sort_index()
y_ld = pdata_ld["dl_tobin"]
x_ld = pdata_ld[["dl_salary", "dl_firm_size", "dl_leverage", "dl_rd_intensity", "dl_cash_ratio", "dl_ceo_duality"]]
mod_ld = PanelOLS(y_ld, x_ld, entity_effects = True, time_effects = True)
res_ld = mod_ld.fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)
print(res_ld.summary)


## One-way Fixed Effect 
g = pdata_ld.groupby(level = 0)
y_ow = g["dl_tobin"].shift(1).rename("L1_dl_tobin")
x_ow = pd.DataFrame({
    "L1_dl_salary"  :g["dl_salary"].shift(1),
    "L1_dl_firm_size"   :g["dl_firm_size"].shift(1),
    "L1_dl_leverage"    :g["dl_leverage"].shift(1),
    "L1_dl_rd_intensity"    :g["dl_rd_intensity"].shift(1),
    "L1_dl_cash_ratio"  :g["dl_cash_ratio"].shift(1),
    "L1_dl_ceo_duality" : g["dl_ceo_duality"].shift(1)
})
panel = pd.concat([y_ow, x_ow], axis = 1).dropna()
y1 = panel["L1_dl_tobin"]
x1 = panel.drop(columns = ["L1_dl_tobin"])
mod_ow = PanelOLS(y1,x1, entity_effects = True)
res_ow_cl = mod_ow.fit(cov_type = "clustered", cluster_entity = True)
print("\n[One-way FE | clustered by entity]")
print(res_ow_cl.summary)

res_ow_dk = mod_ow.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)
print("\n[One-way FE | Driscoll-Kraay (Bartlett,. bw = 2)]")
print(res_ow_dk.summary)

## BG/BP Test (Pooled OLS, non-panel-aware)
x_ols = sm.add_constant(x1, has_constant = "add")
ols_pool = sm.OLS(y1, x_ols).fit()
print("\n[BG via pooled OLS]")
print(acorr_breusch_godfrey(ols_pool, nlags = 2))

bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_pool.resid, ols_pool.model.exog)
print(f"[BP via pooled OLS] LM = {bp_lm:.3f} p = {bp_lm_p:.4g} | F = {bp_f:.3f} p = {bp_f_p:.4g}")

e = res_ow_dk.resids.rename("e").to_frame()
e["e_lag"] = e.groupby(level = 0)["e"].shift(1)
e = e.dropna()
arl_fe = PanelOLS(e["e"], e[["e_lag"]], entity_effects = True).fit(cov_type = "clustered", cluster_entity = True)
print("\n[AR(1) on FE residuals]")
print(arl_fe.summary)


## Fixed Effect + salary + Year dummy ( 2008, 2020, 2021)
pdata_ld = pdata_ld.sort_index()
years = pdata_ld.index.get_level_values("year")
yr_dum = pd.DataFrame({
    "year_2008": (years == 2008).astype(int),
    "year_2020": (years == 2020).astype(int),
    "year_2021": (years == 2021).astype(int),
}, index = pdata_ld.index)
yr_dum = yr_dum.reindex(panel.index)

g = pdata_ld.groupby(level =0)
y = g["dl_tobin"].shift(1).rename("L1_dl_tobin")
x = pd.DataFrame({
    "L1_dl_salary"  :g["dl_salary"].shift(1),
    "L1_dl_firm_size"   :g["dl_firm_size"].shift(1),
    "L1_dl_leverage"    :g["dl_leverage"].shift(1),
    "L1_dl_rd_intensity"    :g["dl_rd_intensity"].shift(1),
    "L1_dl_cash_ratio"  :g["dl_cash_ratio"].shift(1),
    "L1_dl_ceo_duality" :g["dl_ceo_duality"].shift(1),
})
x = pd.concat([x, yr_dum], axis = 1)
panel = pd.concat([y, x], axis = 1 ).dropna()
y_fe = panel["L1_dl_tobin"]
x_fe = panel.drop(columns = ["L1_dl_tobin"])
x1_3yr = x_fe.copy()



mod_fe_3yr = PanelOLS(y_fe, x_fe, entity_effects = True)
res_fe_3yr_cl= mod_fe_3yr.fit(cov_type = "clustered", cluster_entity = True)
print("\n[FE(one-way) + year dummies (2008/2020/2021) | Clustered by entity]")
print(res_fe_3yr_cl.summary)

# Driscoll-Kraay
res_fe_3yr_dk = mod_fe_3yr.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)
print("\n[FE(one-way) + year dummies | Driscoll- Kraay (Bartlett, bw = 2)]")
print(res_fe_3yr_dk.summary)

#BG\BP
x_ols = sm.add_constant(x_fe, has_constant = "add")
ols_pool = sm.OLS(y_fe, x_ols).fit()
print("\n[BG via pooled OLS]")
print(acorr_breusch_godfrey(ols_pool, nlags =2))

bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_pool.resid, ols_pool.model.exog)
print(f"[BP via pooled OLS] LM = {bp_lm:.3f} p = {bp_lm_p:.4g} | F ={bp_f:.3f} p = {bp_f_p:.4g}")


## Twoway Effect 
pdata_ld = pdata_ld.sort_index()
g = pdata_ld.groupby(level = 0)

y = g["dl_tobin"].shift(1).rename("L1_dl_tobin")
x = pd.DataFrame({
    "L1_dl_salary"      : g["dl_salary"].shift(1),
    "L1_dl_firm_size"   : g["dl_firm_size"].shift(1),
    "L1_dl_leverage"    : g["dl_leverage"].shift(1),
    "L1_dl_rd_intensity": g["dl_rd_intensity"].shift(1),
    "L1_dl_cash_ratio"  : g["dl_cash_ratio"].shift(1),
    "L1_dl_ceo_duality" : g["dl_ceo_duality"].shift(1),
})
panel_ld = pd.concat([y, x], axis=1).dropna()
y_ld = panel_ld["L1_dl_tobin"]
x_ld = panel_ld.drop(columns=["L1_dl_tobin"])

mod_tw = PanelOLS(y_ld, x_ld, entity_effects = True, time_effects = True)
res_tw_2clu = mod_tw.fit(cov_type = "clustered", cluster_entity = True, cluster_time = True)
print("\n[Two-way FE | two-way clustered(entity + time)]")
print(res_tw_2clu.summary)

res_tw_dk = mod_tw.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)
print("\n[Two-way FE | Driscoll-Kraay (Bartlett, bw = 2)]")
print(res_tw_dk.summary)

## Random Effects
mod_re = RandomEffects(y_ld, x_ld)
res_re = mod_re.fit()
print("\n[Random Effects | default]")
print(res_re.summary)

res_re_cl = mod_re.fit(cov_type = "clustered", cluster_entity = True)
print("\n[Random Effects | clustered by entity]")
print(res_re_cl.summary)
                       
res_re_dk = mod_re.fit(cov_type = "kernel", kernel = "bartlett", bandwidth = 2)                      
print("\n[Random Effects | Driscoll-Kraay (Bartlett, bw = 2)]")
print(res_re_dk.summary)

##DK
fe_oneway_dk = mod_ow.fit(cov_type="kernel", kernel="bartlett", bandwidth=2)
print("\n[One-way FE | Driscoll–Kraay]")
print(fe_oneway_dk.summary)

##BG/BP
x_ols = sm.add_constant(x_ld, has_constant = "add")
ols_pool = sm.OLS(y_ld, x_ols).fit()

print("\n[BG via pooled OLS] (LM, LM p, F, F p)")
print(acorr_breusch_godfrey(ols_pool, nlags=2))

bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(ols_pool.resid, ols_pool.model.exog)
print(f"[BP via pooled OLS] LM = {bp_lm:.3f} p = {bp_lm_p:.4g} | F = {bp_f:.3f} p = {bp_f_p:.4g}")\


##Hausman_test
fe_oneway_unadj   = PanelOLS(y1, x1, entity_effects=True).fit(cov_type="unadjusted")
fe_3yr_unadj      = PanelOLS(y1, x1_3yr, entity_effects=True).fit(cov_type="unadjusted")
fe_twoway_unadj   = PanelOLS(y1, x1, entity_effects=True, time_effects=True).fit(cov_type="unadjusted")
re_unadj          = RandomEffects(y1, x1).fit(cov_type="unadjusted")

h1 = hausman_test(fe_oneway_unadj, re_unadj, "FE(one-way)", "RE")
h2 = hausman_test(fe_3yr_unadj, fe_oneway_unadj, "FE(one-way + year dummies)", "FE(one-way)")
h3 = hausman_test(fe_twoway_unadj, fe_oneway_unadj, "FE(two-way)", "FE(one-way)")
h4 = hausman_test(fe_twoway_unadj, fe_3yr_unadj, "FE(two-way)", "FE(one-way + year dummies)")

for h in (h1, h2, h3, h4):
    print(f"{h['compared']}: chi2 = {h['stat']:.3f}, df = {h['df']}, p = {h['pval']:.4g}")
    print("  params compared:", h["params_compared"])
