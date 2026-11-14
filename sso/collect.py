import glob,pandas as pd
S=pd.concat([pd.read_csv(f) for f in glob.glob("parts/**/summary_*.csv",recursive=True)])
M=[pd.read_csv(f) for f in glob.glob("parts/**/movers_*.csv",recursive=True)]
M=pd.concat(M) if M else pd.DataFrame()
S.to_csv("all_summary.csv",index=False); M.to_csv("all_movers.csv",index=False)
print(len(S),"observations;",int(S.downloaded.sum()),"downloaded;",len(M),"movers total")
print("median injection recovery",S.inj_recovery.median() if "inj_recovery" in S else "n/a")
