import os
import pandas as pd

from pm4py import fitness_token_based_replay, precision_token_based_replay, fitness_alignments, precision_alignments
import numpy as np
from .eventLog import EventLog
import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def visualize_petri_net(net: pm4py.PetriNet, 
                        initial_marking: pm4py.Marking = None, 
                        final_marking: pm4py.Marking = None,
                        format: str = "png",
                        save_path: str = None,
                        show: bool = True):
    """
    Function to visualize a pm4py PetriNet
    
    Args:
        net: pm4py.PetriNet object
        initial_marking: Initial marking (optional)
        final_marking: Final marking (optional)
        format: Output format ("png", "svg", "pdf", etc.)
        save_path: File path to save (None to skip saving)
        show: If True, display the visualization on screen
    
    Returns:
        gviz: graphviz object
    """
    # Set visualization parameters
    parameters = {
        pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: format
    }
    
    # Create PetriNet visualization
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
    
    # Save to file
    if save_path:
        pn_visualizer.save(gviz, save_path)
        print(f"Petri Net saved to: {save_path}")
    
    # Display on screen
    if show:
        pn_visualizer.view(gviz)
    
    return gviz


def visualize_petri_net_from_dict(net_dict: dict, 
                                   case_id: str = None,
                                   format: str = "png",
                                   save_path: str = None,
                                   show: bool = True):
    """
    Function to visualize PetriNet data in dictionary format
    
    Args:
        net_dict: Dictionary in the format {'net': PetriNet, 'initial_marking': Marking, 'final_marking': Marking}
        case_id: The ID if visualizing a network for a specific Case ID
        format: Output format ("png", "svg", "pdf", etc.)
        save_path: File path to save
        show: If True, display the visualization on screen
    
    Returns:
        gviz: graphviz object
    """
    if case_id and case_id in net_dict:
        net_data = net_dict[case_id]
    else:
        net_data = net_dict
    
    net = net_data['net']
    initial_marking = net_data.get('initial_marking', None)
    final_marking = net_data.get('final_marking', None)
    
    return visualize_petri_net(net, initial_marking, final_marking, format, save_path, show)

def check_conformance(Net:pm4py.PetriNet, Log:EventLog) -> pd.DataFrame:
    result_log = pd.DataFrame(columns=['Case ID', 'Correctness', 'Fitness', 'Precision', 'F1 Score'])

    caseid=Log['Case ID'].unique()[0]
    if caseid not in Net:
        raise ValueError(f"Case ID {caseid} not found in Net.")

    reason_net = Net[caseid]['net']
    reason_im = Net[caseid]['initial_marking']
    reason_fm = Net[caseid]['final_marking']
    correctness = Net[caseid].get('correctness', None)

    fitness = fitness_alignments(Log, reason_net, reason_im, reason_fm,
                                     activity_key='Activity', timestamp_key='Step', case_id_key='Case ID')['log_fitness']

    precision = precision_alignments(Log, reason_net, reason_im, reason_fm,
                                    activity_key='Activity', timestamp_key='Step', case_id_key='Case ID')

    f1_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0

    new_row = {
        'Case ID': caseid, 'Correctness': correctness,
        'Fitness': fitness, 'Precision': precision,
        'F1 Score': f1_score}
    result_log = pd.concat([result_log, pd.DataFrame([new_row])], ignore_index=True)

    return result_log


class Checker:
    def __init__(self, TrueLog: object, ReasonNet: dict):

        self.truelog = TrueLog.log
        self.check_log()
        self.net = ReasonNet

    def check_log(self):
        assert self.truelog['Step'].dtype == 'datetime64[ns]', f'datetime expected, but got {self.truelog["Step"].dtype}'
        assert 'Case ID' in self.truelog.columns, f'Case ID column not found in log'
        assert 'Activity' in self.truelog.columns, f'Activity column not found in log'
        assert 'Step' in self.truelog.columns, f'Step column not found in log'

    def check(self):
        check=check_conformance(self.net, self.truelog)
        return check

    def visualize(self, 
                  case_id: str = None,
                  format: str = "png",
                  save_path: str = None,
                  show: bool = True):
        """
        Method to visualize the PetriNet stored in Checker
        
        Args:
            case_id: The ID if visualizing a network for a specific Case ID
                     If None, uses the first Case ID from truelog
            format: Output format ("png", "svg", "pdf", etc.)
            save_path: File path to save (None to skip saving)
            show: If True, display the visualization on screen
        
        Returns:
            gviz: graphviz object
        """
        if case_id is None:
            case_id = self.truelog['Case ID'].unique()[0]
        
        if case_id not in self.net:
            raise ValueError(f"Case ID {case_id} not found in Net.")
        
        net_data = self.net[case_id]
        net = net_data['net']
        initial_marking = net_data.get('initial_marking', None)
        final_marking = net_data.get('final_marking', None)
        
        return visualize_petri_net(net, initial_marking, final_marking, format, save_path, show)

    def visualize_all(self,
                      format: str = "png",
                      save_dir: str = None,
                      show: bool = False):
        """
        Method to visualize all PetriNets stored in Checker
        
        Args:
            format: Output format ("png", "svg", "pdf", etc.)
            save_dir: Directory path to save (None to skip saving)
            show: If True, display the visualization on screen
        
        Returns:
            dict: Dictionary in the format {case_id: gviz}
        """
        results = {}
        for case_id in self.net.keys():
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{case_id}.{format}")
            
            results[case_id] = self.visualize(case_id, format, save_path, show)
        
        return results