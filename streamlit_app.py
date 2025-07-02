import streamlit as st
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

st.set_page_config(page_title="Workforce Scheduler", layout="wide")
st.title("Workforce Scheduling Optimization App")

st.sidebar.header("Step 1: Upload Your Data")
emp_file = st.sidebar.file_uploader("Employee Data (CSV)", type="csv")
demand_file = st.sidebar.file_uploader("Demand Data (CSV)", type="csv")
avail_file = st.sidebar.file_uploader("Availability Data (CSV)", type="csv")
constr_file = st.sidebar.file_uploader("Constraints Data (CSV)", type="csv")

if st.sidebar.button("Run Optimization"):
    if not all([emp_file, demand_file, avail_file, constr_file]):
        st.error("Please upload all required files.")
    else:
        employees_df = pd.read_csv(emp_file)
        demand_df = pd.read_csv(demand_file)
        availability_df = pd.read_csv(avail_file)
        constraints_df = pd.read_csv(constr_file)

        employee_ids = employees_df["EmployeeID"].tolist()
        shifts = ["Morning", "Afternoon", "Night"]
        dates = pd.to_datetime(demand_df["Date"]).dt.date.unique().tolist()
        roles = demand_df["Role"].unique().tolist()

        emp_role_skill = employees_df.set_index("EmployeeID")[["Role", "Skill"]].to_dict("index")
        availability = availability_df.set_index(["EmployeeID", "Date", "Shift"])["Available"].to_dict()
        demand = demand_df.set_index(["Date", "Shift", "Role"])["RequiredCount"].to_dict()
        constraints = constraints_df.set_index("EmployeeID").to_dict("index")

        assign = {
            (e, d, s): LpVariable(f"assign_{e}_{d}_{s}", cat=LpBinary)
            for e in employee_ids for d in dates for s in shifts
        }

        model = LpProblem("WorkforceScheduling", LpMinimize)
        model += lpSum(assign[e, d, s] for e in employee_ids for d in dates for s in shifts)

        for d in dates:
            for s in shifts:
                for r in roles:
                    eligible_emps = [e for e in employee_ids if emp_role_skill[e]["Role"] == r]
                    model += lpSum(assign[e, d, s] for e in eligible_emps) >= demand.get((d, s, r), 0)

        for e in employee_ids:
            for d in dates:
                model += lpSum(assign[e, d, s] for s in shifts) <= 1

        for e in employee_ids:
            off_day = constraints[e]["WeeklyOffDay"]
            for d in dates:
                if pd.to_datetime(d).strftime("%A") == off_day:
                    for s in shifts:
                        model += assign[e, d, s] == 0

        for e in employee_ids:
            for d in dates:
                for s in shifts:
                    if availability.get((e, str(d), s), 0) == 0:
                        model += assign[e, d, s] == 0

        for e in employee_ids:
            max_hours = constraints[e]["MaxHoursPerWeek"]
            model += lpSum(assign[e, d, s] * 8 for d in dates for s in shifts) <= max_hours

        model.solve()

        roster = []
        for (e, d, s), var in assign.items():
            if var.varValue == 1:
                roster.append({
                    "EmployeeID": e,
                    "Date": d,
                    "Shift": s,
                    "Role": emp_role_skill[e]["Role"],
                    "Skill": emp_role_skill[e]["Skill"]
                })

        roster_df = pd.DataFrame(roster)
        st.success("Optimization completed successfully!")
        st.dataframe(roster_df)

        csv = roster_df.to_csv(index=False)
        st.download_button("Download Roster as CSV", data=csv, file_name="optimized_roster.csv", mime="text/csv")
