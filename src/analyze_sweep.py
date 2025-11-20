import wandb
import pandas as pd

api = wandb.Api()
# sweep = api.sweep("ashishreddytummurinyu/MC-ExOpt-Sweep/sweeps/jtinpd6e") 
# The above fails because jtinpd6e is a RUN ID, not a SWEEP ID.
# We can just query all runs in the project.

# Let's just get runs from the project and sort by metric.
runs = api.runs("ashishreddytummurinyu/MC-ExOpt-Sweep")

summary_list = [] 
config_list = [] 
name_list = [] 
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to ensure we get a dict of values
    summary_list.append(run.summary._json_dict) 
    
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items() if not k.startswith('_')}
    ) 
    
    name_list.append(run.name)       

runs_df = pd.DataFrame({
    "name": name_list,
    "improvement": [s.get("val_improvement_pct") for s in summary_list],
    "config": config_list
})

# Sort by improvement (descending, since higher is better, even if negative)
runs_df = runs_df.sort_values("improvement", ascending=False)

print("Top 5 Runs:")
for i, row in runs_df.head(5).iterrows():
    print(f"Name: {row['name']}, Improvement: {row['improvement']}%")
    print(f"Config: {row['config']}")
    print("-" * 20)
