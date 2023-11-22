import numpy as np

spo2list = ['spo2','spo2_v2','spo2_v3','spo2_v4','spo2_v5','spo2_v6', 'spo2_v7','spo2_v8','spo2_v9','spo2_v10']
so2list = ['so2','so2_v2','so2_v3','so2_v4','so2_v5','so2_v6', 'so2_v7','so2_v8','so2_v9','so2_v10']
so2list_research = ['research_gem_so2','research_gem_so2_v2','research_gem_so2_v3','research_gem_so2_v4','research_gem_so2_v5','research_gem_so2_v6','research_gem_so2_v7','research_gem_so2_v8','research_gem_so2_v9','research_gem_so2_v10']
collectionreasonlist = ['abg_collection_reason',
'abg_collection_reason_v2',
'abg_collection_reason_v3',
'abg_collection_reason_v4',
'abg_collection_reason_v5',
'abg_collection_reason_v6',
'abg_collection_reason_v7',
'abg_collection_reason_v8',
'abg_collection_reason_v9',
'abg_collection_reason_v10']
artmaplist = ['art_map', 'art_map_v2', 'art_map_v3','art_map_v4','art_map_v5', 'art_map_v6', 'art_map_v7', 'art_map_v8','art_map_v9','art_map_v10']
stabilitylist = ['so2_period_of_stability', 'spo2_period_of_stability_v2', 'spo2_period_of_stability_v3', 'spo2_period_of_stability_v4', 'spo2_period_of_stability_v5', 'spo2_period_of_stability_v6', 'spo2_period_of_stability_v7', 'spo2_period_of_stability_v8', 'spo2_period_of_stability_v9', 'spo2_period_of_stability_v10']
pilist = ['perfusion', 'perfusion_v2', 'perfusion_v3', 'perfusion_v4', 'perfusion_v5',
'perfusion_v6', 'perfusion_v7', 'perfusion_v8', 'perfusion_v9', 'perfusion_v10']
massimo_pi_list = ['masimo_perfusion', 'masimo_perfusion_v2', 'masimo_perfusion_v3', 'masimo_perfusion_v4', 'masimo_perfusion_v5',
'masimo_perfusion_v6', 'masimo_perfusion_v7', 'masimo_perfusion_v8', 'masimo_perfusion_v9', 'masimo_perfusion_v10']
capture_time_list = ['capture_time','capture_time_v2','capture_time_v3','capture_time_v4','capture_time_v5',
'capture_time_v6','capture_time_v7','capture_time_v8','capture_time_v9','capture_time_v10']
sampleanalysis_time_list = ['time_of_so2_sample_analysi','so2_time_v2','so2_time_v3','so2_time_v4','so2_time_v5',
'so2_time_v6','so2_time_v7','so2_time_v8','so2_time_v9','so2_time_v10']
probelocation_list = ['probe_location', 'probe_location_v2', 'probe_location_v3', 'probe_location_v4', 'probe_location_v5', 
                        'probe_location_v6', 'probe_location_v7', 'probe_location_v8', 'probe_location_v9', 'probe_location_v10']

msoldlist = ['ms_inner_arm','ms_fingernail','ms_surface_b','ms_surface_c','ms_forehead']
msnewlist = ['ms_new_inner_arm','ms_new_fingernail','ms_new_dorsal','ms_new_ventral','ms_new_forehead']

mslocations = ['Inner Arm','Fingernail','Dorsal','Ventral','Forehead']
monkvalues = ['A','B','C','D','E','F','G','H','I','J']
vlvalues = ["Light (1-15)", "Light Medium (16-21)","Dark Medium (22-28)","Dark (29-36)"]

vllist = ['vl_inner_arm','vl_fingernail','vl_surface_b','vl_surface_c']


def vlbins(row, var):
    try: 
        float(row[var])
        if row[var] >= 1 and row[var] <=15:
            return "Light (1-15)"
        elif row[var] >= 16 and row[var] <=21:
            return "Light Medium (16-21)"
        elif row[var] >= 22 and row[var] <=28:
            return "Dark Medium (22-28)"
        elif row[var] >= 29 and row[var] <= 36:
            return "Dark (29-36)"
        else:
            return np.nan
    except ValueError:
        return row[var]