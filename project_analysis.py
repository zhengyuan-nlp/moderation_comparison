import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from openpyxl import Workbook
import configparser
# import seaborn as sns

params = {'legend.fontsize': 'large',
          'figure.figsize': (12.30,5.11),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

def openFile(filepath=""):
    if filepath == "":
        print("File path not set, check config file")
        exit()
    data = pd.read_csv(filepath)
    return data


def create_save_file(filepath):
    try: 
        os.mkdir(filepath) 
    except OSError as error: 
        pass

def linear_supervisors_vs_moderator_plot(full_data,save_file_path):
    global GRADE_BOUNDARIES
    global GRADE

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    scatter_colors = []
    if GRADE == 'BSC':
        for i in full_data["Final Mark"]:
            if i >= GRADE_BOUNDARIES["First"]:
                scatter_colors.append(mcolors.CSS4_COLORS["orange"])
            elif i >= GRADE_BOUNDARIES["Second"]:
                scatter_colors.append(mcolors.CSS4_COLORS["lightgreen"])
            elif i >= GRADE_BOUNDARIES["Third"]:
                scatter_colors.append(mcolors.CSS4_COLORS["slateblue"])
            else:
                scatter_colors.append(mcolors.CSS4_COLORS["gold"])
    elif GRADE == 'MSC':
        for i in full_data["Final Mark"]:
            if i >= GRADE_BOUNDARIES["Distinction"]:
                scatter_colors.append(mcolors.CSS4_COLORS["orange"])
            elif i >= GRADE_BOUNDARIES["Merit"]:
                scatter_colors.append(mcolors.CSS4_COLORS["lightgreen"])
            elif i >= GRADE_BOUNDARIES["Pass"]:
                scatter_colors.append(mcolors.CSS4_COLORS["slateblue"])
            else:
                scatter_colors.append(mcolors.CSS4_COLORS["gold"])

    else:
        print("Grade not recognised")
        return
    sup_grade = np.array(full_data["First Calculated Mark"]).reshape(-1,1)
    mod_grade = np.array(full_data["Moderation Outcome Mark"]).reshape(-1,1)

    sup_mean = round(np.mean(sup_grade),1)
    mod_mean = round(np.mean(mod_grade),1)

    combined_grade = np.concatenate((sup_grade,mod_grade),axis=1)
    occurance = {}
    for n in combined_grade:
        if (f"{n}") not in occurance:
            occurance[f"{n}"] = 0.5
        else:
            occurance[f"{n}"] += 1

    blob_size = []
    for n in combined_grade:
        blob_size.append(occurance[f"{n}"] * 50)

    ax.scatter(sup_grade.reshape(-1), mod_grade.reshape(-1),c=scatter_colors,s=blob_size)
    ax.set_ylabel('Moderator')
    ax.set_xlabel('Supervisor')
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)

    z = np.polyfit(sup_grade.reshape(-1),mod_grade.reshape(-1),1)
    p = np.poly1d(z)
    x = [0,100]

    ax.plot(x,p(x),mcolors.CSS4_COLORS["slategray"],linestyle='dashed')
    
    txt = f"Moderator average: {mod_mean}; Supervisor average: {sup_mean}"
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    
    plt.xticks(range(0,100,10))
    plt.yticks(range(0,100,10))
    plt.grid()
    custom = [Line2D([], [], marker='.', markersize=20, color=mcolors.CSS4_COLORS["orange"], linestyle='None'),
             Line2D([], [], marker='.', markersize=20, color=mcolors.CSS4_COLORS["lightgreen"], linestyle='None'),
             Line2D([], [], marker='.', markersize=20, color=mcolors.CSS4_COLORS["slateblue"], linestyle='None'),
             Line2D([], [], marker='.', markersize=20, color=mcolors.CSS4_COLORS["gold"], linestyle='None')]

    list_grades = list(GRADE_BOUNDARIES.keys())[::-1]
    list_grades.append('Fail')
    plt.legend(custom, list_grades)
    save_file_path = os.path.join(save_file_path, "linear_supervisors_vs_moderator_plot.png")
    plt.savefig(save_file_path)
    # plt.show()



def moderator_moderation_table(full_data,save_file_path, filter1, filter2):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

    
    empty_array = []
    for index, row in full_data.iterrows():
        if row['First Calculated Mark'] != row['Moderation Outcome Mark']:
            empty_array.append([row['Moderator Name'],row['Supervisor Name'],row['Student Number'],
                                row['First Calculated Mark'],row['Moderation Outcome Mark'],
                                row['Final Mark'], row['Final Mark'] -  row['First Calculated Mark'],
                                row['Moderation Outcome Mark'] - row['First Calculated Mark']])

    table_df = pd.DataFrame(columns = ["Moderator","Supervisor","Student ID", "1st mark","2nd mark","Final mark","Final-1st","2nd-1st"], data = empty_array)
    table_df.sort_values(by="Moderator", inplace=True)
    
    ax.axis('off')
    ax.axis('tight')
    filtered_table_df = table_df.loc[(table_df['Final-1st'] >= filter1) | (table_df['2nd-1st'] >= filter1)]
    table_1 = ax.table(cellText=filtered_table_df.values,colLabels=table_df.columns,loc='center')
    table_1.scale(1,1) 
    fig.tight_layout()
    save_file_path_1 = os.path.join(save_file_path, "moderator_moderation_table.png")
    plt.savefig(save_file_path_1)
    plt.cla()


    supervisors = table_df['Supervisor'].unique()
    pairs = {}
    for supervisor in supervisors:
        pairs[supervisor] = table_df.loc[table_df['Supervisor'] == supervisor]['Moderator'].unique()

    empty_array_2 = []
    for supervisor in supervisors:
        for moderator in pairs[supervisor]:
            temp_pair_df = table_df.loc[(table_df['Supervisor'] == supervisor) & (table_df['Moderator'] == moderator)]
            empty_array_2.append([moderator,supervisor,round(temp_pair_df['2nd-1st'].mean(),2),
                                  round(abs(temp_pair_df['2nd-1st'].mean()),2), round(temp_pair_df['2nd-1st'].abs().mean(),2)])

    table_df_2 = pd.DataFrame(columns = ["Moderator","Supervisor","avg modulated","avg abs modulated", "abs avg modulated"], data= empty_array_2)

    ax.axis('off')
    ax.axis('tight')
    filtered_table_df_2 = table_df_2.loc[(table_df_2["avg abs modulated"] >= filter2) | (table_df_2["abs avg modulated"] >= filter2)]
    table_2 = ax.table(cellText=filtered_table_df_2.values,colLabels=filtered_table_df_2.columns,loc='center')
    table_2.scale(1,0.65) 
    table_2.set_fontsize(6)
    fig.tight_layout()
    save_file_path_2 = os.path.join(save_file_path, "moderator_moderation_table_stats.png")
    plt.savefig(save_file_path_2)
    plt.cla()


    moderators = table_df['Moderator'].unique()
    empty_array_3 = []
    for moderator in moderators:
        single_moderator_df = table_df.loc[table_df['Moderator'] == moderator]
        empty_array_3.append([moderator, single_moderator_df["2nd-1st"].max(), single_moderator_df["2nd-1st"].min()])

    table_df_3 = pd.DataFrame(columns = ["Moderator","Max","Min"], data= empty_array_3)

    ax.axis('off')
    ax.axis('tight')
    table_3 = ax.table(cellText=table_df_3.values,colLabels=table_df_3.columns,loc='center')
    table_3.scale(1,1) 
    fig.tight_layout()
    save_file_path_3 = os.path.join(save_file_path, "moderator_max_min.png")
    plt.savefig(save_file_path_3)



    return table_df, filtered_table_df, table_df_2, filtered_table_df_2, table_df_3

def moderation_mark_change_bin(data,save_file_path):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.2)

    sup_grade = np.array(full_data["First Calculated Mark"]).reshape(-1,1)
    mod_grade = np.array(full_data["Moderation Outcome Mark"]).reshape(-1,1)
    final_grade = np.array(full_data["Final Mark"]).reshape(-1,1)

    mark_changes = final_grade - sup_grade

    _, counts = np.unique(mark_changes, return_counts=True)

    bins = np.arange(int(np.min(mark_changes)), int(np.max(mark_changes))+1, 1)
    plt.yticks(range(0,int(np.max(counts)),10))
    xlabels = []
    for n in bins:
        xlabels.append([n, n+1])

    xtick_loc = np.arange(np.min(mark_changes),np.max(mark_changes))
    # plt.tick_params(axis='x',which='both',bottom=False)
    plt.xticks(xtick_loc, xlabels[:-1],rotation=45,fontsize=9)
    plt.xlabel("Difference = Final mark - 1st mark")
    plt.ylabel("No. of projects")
    plt.grid(axis='y')
    plt.hist(mark_changes, bins=bins, align="left")
    save_file_path = os.path.join(save_file_path, "moderation_mark_change_bin.png")
    plt.savefig(save_file_path)
    # plt.show()

def mark_overview(data, save_file_path):
    global GRADE_BOUNDARIES
    global GRADE
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    fig.patch.set_visible(False)

    input_data = pd.DataFrame(full_data, columns=['Final Mark'])
    total_projects = len(input_data.index)
    
    df_to_plot = pd.DataFrame(columns=["Grade","Grade boundaries","No. of projects","%","cumulative %"])

    if GRADE == 'BSC':
        g_1 = input_data[input_data['Final Mark'] >= 70]
        g_2 = input_data.loc[(input_data['Final Mark'] >= 60) & (input_data['Final Mark'] < 70)]
        g_3 = input_data.loc[(input_data['Final Mark'] >= 50) & (input_data['Final Mark'] < 60)]
        g_4 = input_data.loc[(input_data['Final Mark'] >= 40) & (input_data['Final Mark'] < 50)]
        fail = input_data[input_data['Final Mark'] < 40]
        min_fail = int(fail.min()[0])
        max_fail = int(fail.max()[0])
        
        df_to_plot["Grade"] = ["First","Upper second 2.1","Lower seconds 2.2","Third","Fail"]
        df_to_plot["Grade boundaries"] = ["[70,100]","[60,70)","[50,60)","[40,50)",f"(0,40) [{min_fail}, {max_fail}]"]
        df_to_plot["No. of projects"] = [len(g_1.index),len(g_2.index),len(g_3.index),len(g_4.index),len(fail.index)]
    
    elif GRADE == 'MSC':
        g_1 = input_data[input_data['Final Mark'] >= 70]
        g_2 = input_data.loc[(input_data['Final Mark'] >= 60) & (input_data['Final Mark'] < 70)]
        g_3 = input_data.loc[(input_data['Final Mark'] >= 50) & (input_data['Final Mark'] < 60)]
        fail = input_data[input_data['Final Mark'] < 50]
        min_fail = int(fail.min()[0])
        max_fail = int(fail.max()[0])
        
        df_to_plot["Grade"] = ["Distinction","Merit","Pass","Fail"]
        df_to_plot["Grade boundaries"] = ["[70,100]","[60,70)","[50,60)",f"(0,40) [{min_fail}, {max_fail}]"]
        df_to_plot["No. of projects"] = [len(g_1.index),len(g_2.index),len(g_3.index),len(fail.index)]
            
    else:
        print("Grade not recognised")
        return None
    
    percentages = [((n/total_projects) * 100) for n in df_to_plot["No. of projects"]]
    df_to_plot["%"] = [round(n,2) for n in percentages]
    df_to_plot["cumulative %"] = [round(sum(percentages[:n+1]),2) for n in range(len(df_to_plot["Grade"]))]




    ax.axis('off')
    ax.axis('tight')

    ytable = ax.table(cellText=df_to_plot.values,colLabels=df_to_plot.columns,loc='center')
    ytable.scale(1,5) 
    fig.tight_layout()
    save_file_path = os.path.join(save_file_path, "mark_overview.png")
    plt.savefig(save_file_path)
    return df_to_plot

    # plt.show()


def mark_overview_bin(data, save_file_path):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.2)

    final_grade = np.array(full_data["Final Mark"]).reshape(-1,1)

    _, counts = np.unique(final_grade, return_counts=True)

    bins = np.arange(int(np.min(final_grade)), int(np.max(final_grade))+1, 1)
    plt.yticks(range(0,int(np.max(counts)),10))
    xlabels = []
    for n in bins:
        xlabels.append([n, n+1])

    xtick_loc = np.arange(np.min(final_grade),np.max(final_grade))
    # plt.tick_params(axis='x',which='both',bottom=False)
    plt.xticks(xtick_loc, xlabels[:-1],rotation=80,fontsize=8)
    plt.xlabel("Mark overview")
    plt.ylabel("No. of projects")
    plt.grid(axis='y')
    plt.hist(final_grade, bins=bins, align="left")
    save_file_path = os.path.join(save_file_path, "mark_overview_bin.png")
    plt.savefig(save_file_path)



def replace_mark_with_grade(i):
    global GRADE_BOUNDARIES
    global GRADE
    
    if GRADE == 'BSC':
        if i >= GRADE_BOUNDARIES["First"]:
            return "First"
        elif i >= GRADE_BOUNDARIES["Second"]:
            return "Second"
        elif i >= GRADE_BOUNDARIES["Third"]:
            return "Third"
        else:
            return "Fail"
    elif GRADE == 'MSC':
        if i >= GRADE_BOUNDARIES["Distinction"]:
            return "Distinction"
        elif i >= GRADE_BOUNDARIES["Merit"]:
            return "Merit"
        elif i >= GRADE_BOUNDARIES["Pass"]:
            return "Pass"
        else:
            return "Fail"
    else:
        return

def class_change_after_moderation(data, save_file_path):
    global GRADE
    global GRADE_BOUNDARIES

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    fig.patch.set_visible(False)

    input_data = pd.DataFrame(full_data, columns=['Student Number','First Calculated Mark','Moderation Outcome Mark','Final Mark'])

    boundary = list(GRADE_BOUNDARIES.keys())


    input_data['First Calculated Grade'] = input_data['First Calculated Mark'].apply(replace_mark_with_grade)
    input_data['Final Grade'] = input_data['Final Mark'].apply(replace_mark_with_grade)

    g1_to_g2 = input_data.loc[(input_data['First Calculated Grade'] == boundary[2]) & (input_data['Final Grade'] == boundary[1])]

    g2_to_g3 = input_data.loc[(input_data['First Calculated Grade'] == boundary[1]) & (input_data['Final Grade'] == boundary[0])]

    g3_to_fail = input_data.loc[(input_data['First Calculated Grade'] == boundary[0]) & (input_data['Final Grade'] == 'Fail')]

    g2_to_g1 = input_data.loc[(input_data['First Calculated Grade'] == boundary[1]) & (input_data['Final Grade'] == boundary[2])]

    g3_to_g2 = input_data.loc[(input_data['First Calculated Grade'] == boundary[0]) & (input_data['Final Grade'] == boundary[1])]

    fail_to_g3 = input_data.loc[(input_data['First Calculated Grade'] == 'Fail') & (input_data['Final Grade'] == boundary[0])]


    df_to_plot = pd.DataFrame(columns=["Grade Change", "No. of projects", "Student ID (Final mark - 1st mark)"])

    df_to_plot["Grade Change"] = [f"{boundary[2]} to {boundary[1]}",f"{boundary[1]} to {boundary[0]}",f"{boundary[0]} to Fail",\
                                  f"{boundary[1]} to {boundary[2]}",f"{boundary[0]} to {boundary[1]}",f"Fail to {boundary[0]}"]

    df_to_plot["No. of projects"] = [len(g1_to_g2.index),len(g2_to_g3.index),len(g3_to_fail.index), \
                                     len(g2_to_g1.index),len(g3_to_g2.index),len(fail_to_g3.index)]


    final_result_list = ([', '.join(g1_to_g2["Student Number"].values.tolist()),', '.join(g2_to_g3["Student Number"].values.tolist()),\
                          ', '.join(g3_to_fail["Student Number"].values.tolist()),', '.join(g2_to_g1["Student Number"].values.tolist()),\
                          ', '.join(g3_to_g2["Student Number"].values.tolist()),', '.join(fail_to_g3["Student Number"].values.tolist())])
    
    df_to_plot["Student ID (Final mark - 1st mark)"] = final_result_list
    ax.axis('off')
    ax.axis('tight')

    ytable = ax.table(cellText=df_to_plot.values,colLabels=df_to_plot.columns,loc='center',colWidths=[0.1,0.1,0.8])
    ytable.scale(1,5) 
    fig.tight_layout()
    save_file_path = os.path.join(save_file_path, "class_change_after_moderation.png")
    plt.savefig(save_file_path)
    return df_to_plot
    # plt.show()

def plot_plots(data, save_file_path,filter1,filter2):
    linear_supervisors_vs_moderator_plot(data,save_file_path)
    mark_overview_bin(data, save_file_path)
    moderation_mark_change_bin(data,save_file_path)
    mod_sup_output, mod_sup_filter_output, mod_sup_stats_output, mod_sup_stats_filter_output, \
        mod_sup_maxmin_output = moderator_moderation_table(data,save_file_path,filter1,filter2)
    class_change_after_moderation_output = class_change_after_moderation(data,save_file_path)
    mark_overview_output = mark_overview(data, save_file_path)
    export_to_excel(class_change_after_moderation_output,mark_overview_output,mod_sup_output, \
                    mod_sup_filter_output, mod_sup_stats_output, mod_sup_stats_filter_output, \
                    mod_sup_maxmin_output,save_file_path)



def export_to_excel(data1, data2, data3, data4, data5, data6, data7, save_file_path):
    save_file_path = os.path.join(save_file_path, "output.xlsx")
    with pd.ExcelWriter(save_file_path) as writer:
        data1.to_excel(writer, sheet_name = 'Class change after moderation')
        data2.to_excel(writer, sheet_name = 'Mark overview')
        data3.to_excel(writer, sheet_name = 'Mod Sup diff for each student')
        data4.to_excel(writer, sheet_name = 'Mod Sup diff filtered')
        data5.to_excel(writer, sheet_name = 'Mod Sup moderation diff')
        data6.to_excel(writer, sheet_name = 'Mod Sup moderation diff filter')
        data7.to_excel(writer, sheet_name = 'Mod max min')



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    boundary_msc = {"Pass": 50, "Merit": 60, "Distinction": 70}
    boundary_bsc = {"Third": 40, "Second": 50, "First": 70}


    FILE_PATH = config.get('project_analysis','csv_file')
    OUTPUT_FOLDER_PATH = config.get('project_analysis','output_folder_path')
    OUTPUT_FOLDER_NAME = config.get('project_analysis','output_folder_name')
    if OUTPUT_FOLDER_NAME == "":
        print("Output folder name not set, please check config file")
        exit()

    GRADE = config.get('project_analysis','grade_boundary')
    if GRADE.upper() == 'BSC':
        GRADE_BOUNDARIES = boundary_bsc
        print("huh")
    elif GRADE.upper() == 'MSC':
        GRADE_BOUNDARIES = boundary_msc
    else:
        print("Grade boundary config not recognised, please check config file")
        exit()

    filter1 = int(config.get('project_analysis','filter1'))
    filter2 = int(config.get('project_analysis','filter2'))

    full_data = openFile(FILE_PATH)
    full_data = full_data.dropna(subset=['First Calculated Mark','Moderation Outcome Mark'])
    save_file_path = os.path.join(OUTPUT_FOLDER_PATH,OUTPUT_FOLDER_NAME)
    create_save_file(save_file_path)

    print(f"Running with {GRADE.upper()} grades")
    print("Grade boundary:", GRADE_BOUNDARIES)
    plot_plots(full_data, save_file_path,filter1,filter2)
    print("Done")
