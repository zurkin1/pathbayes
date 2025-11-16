'''
Maximum flow treats the graph as a network:
- Each edge has a capacity (UDP expression level).
- Sources are starting points; sinks are ending points.
- Maximum flow computes the maximum amount of "signal" that can flow from source to sink.
- This directly models pathway activity as signal propagation capacity.
'''
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import pandas as pd
import networkx as nx
import warnings
from metrics import *
import multiprocessing as mp
from tqdm import tqdm


warnings.simplefilter("error", RuntimeWarning) #Stop on warnings.


def parse_pathway_interactions(relations_file):
    """
    Parse interactions and assign unique IDs to each
    Load and parse interactions into simple pathway_interactions dictionary data structure. e.g.
    pathway_interactions['Adherens junction'][0] = (['baiap2', 'wasf2', 'wasf3', 'wasf1'], 'activation', ['actb', 'actg1'], 'Adherens junction')
    """
    pathway_relations = pd.read_csv(relations_file)
    pathway_relations['source'] = pathway_relations['source'].fillna('').astype(str).str.lower().str.split('*')
    pathway_relations['target'] = pathway_relations['target'].fillna('').astype(str).str.lower().str.split('*')   
    
    interactions_by_pathway = {}
    for idx, row in pathway_relations.iterrows():
        pathway = row['pathway']
        if pathway not in interactions_by_pathway:
            interactions_by_pathway[pathway] = []
        
        # Store interaction with its global ID for fast lookup
        interactions_by_pathway[pathway].append({
            'id': idx,
            'source': row['source'],
            'type': row['interactiontype'],
            'target': row['target'],
            'pathway': pathway
        })
    
    return interactions_by_pathway


def build_pathway_graph_structure(interactions):
    """
    Build the static graph structure for a pathway. Stores only topology and gene names no sample-specific data.
    Returns: NetworkX graph with:
    - Nodes: interaction IDs
    - Node attrs: source_genes, target_genes, interaction_type
    - Edge attrs: genes (the shared genes creating this edge)
    - Graph attrs: corridors (list of (source, sink) tuples for top 20 longest paths)
    """
    G = nx.DiGraph()
    
    # Add all nodes first
    for interaction in interactions:
        i_id = interaction['id']
        G.add_node(
            i_id,
            source_genes=interaction['source'],
            target_genes=interaction['target'],
            interaction_type=interaction['type']
        )
    
    # Create edges based on gene sharing (target of i1 → source of i2)
    for int1 in interactions:
        for int2 in interactions:
            if int1['id'] == int2['id']:
                continue
            
            shared_genes = set(int1['target']) & set(int2['source'])
            if shared_genes and shared_genes != {''}:
                # Store all genes that create this connection
                if G.has_edge(int1['id'], int2['id']):
                    # Add to existing gene list
                    G[int1['id']][int2['id']]['genes'].update(shared_genes)
                else:
                    # Create new edge
                    G.add_edge(int1['id'], int2['id'], genes=shared_genes)

    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks   = [node for node in G.nodes if G.out_degree(node) == 0]
    corridors = []
    if sources and sinks:
        # Make temporary acyclic copy for longest path finding.
        G_temp = G.copy()
        while True:
            try:
                cycle = nx.find_cycle(G_temp, orientation='original')
                G_temp.remove_edge(*cycle[0][:2])
            except nx.exception.NetworkXNoCycle:
                break
        
        # Find top 20 longest paths on the temporary DAG
        for _ in range(20):
            try:
                path = nx.dag_longest_path(G_temp)
            except (nx.NetworkXError, nx.NetworkXNotImplemented):
                break
            if len(path) < 2:
                break
            corridors.append((path[0], path[-1]))
            G_temp.remove_edges_from(list(zip(path, path[1:])))
            if G_temp.number_of_edges() == 0:
                break
    
    G.graph['corridors'] = corridors
    return G


def is_inhibitory(interaction_type):
    """Check if interaction type is inhibitory"""
    inhibitory_keywords = ['inhibition', 'repression', 'dissociation', 'dephosphorylation', 'ubiquitination']
    return any(keyword in interaction_type.lower() for keyword in inhibitory_keywords)


def shallow_fallback_activity(G, sample_udp):
    """
    Old-style node belief: gaussian scaling of sum(incoming UDP) and sum(outgoing UDP)
    Returns average over nodes.
    """
    vals = []
    for node in G.nodes:
        src_genes = G.nodes[node].get("source_genes", [])
        tgt_genes = G.nodes[node].get("target_genes", [])

        src_sum = sum(sample_udp.get(g, 0.0) for g in src_genes)
        tgt_sum = sum(sample_udp.get(g, 0.0) for g in tgt_genes)
        tgt_sum = max(tgt_sum, 1e-10)

        b = consistency_scaling(src_sum, tgt_sum)
        if is_inhibitory(G.nodes[node].get("interaction_type", "")):
            b = -b
        vals.append(b)

    return float(np.mean(vals)) if vals else 0.0


def compute_pathway_flow(G, corridors, sample_udp):
    """
    Compute maximum flow for a pathway using supersource and supersink.
    
    Creates a temporary graph with:
    - All original nodes and edges with UDP-weighted capacities
    - A supersource connected to all corridor sources (infinite capacity)
    - A supersink connected from all corridor sinks (infinite capacity)
    
    Returns the maximum flow value from supersource to supersink.
    """
    # Create a copy of the graph to add supersource/supersink
    G_flow = G.copy()
    
    # Add edge capacities based on UDP values, remove zero-capacity edges
    edges_to_remove = []
    for u, v, data in G_flow.edges(data=True):
        genes = data['genes']
        # Sum UDP values for all genes creating this edge
        total_capacity = sum(sample_udp.get(gene, 0.0) for gene in genes)
        if total_capacity > 0:
            G_flow[u][v]['capacity'] = total_capacity
        else:
            # Mark zero-capacity edges for removal
            edges_to_remove.append((u, v))
    
    # Remove zero-capacity edges to avoid numerical issues
    G_flow.remove_edges_from(edges_to_remove)
    
    # Extract unique sources and sinks from corridors
    sources = list(set(src for src, _ in corridors))
    sinks = list(set(snk for _, snk in corridors))
    
    # Add supersource and supersink nodes
    supersource = 'SUPERSOURCE'
    supersink = 'SUPERSINK'
    
    G_flow.add_node(supersource)
    G_flow.add_node(supersink)
    
    # Connect supersource to all corridor sources with infinite capacity
    for source in sources:
        G_flow.add_edge(supersource, source, capacity=1e9)
    
    # Connect all corridor sinks to supersink with infinite capacity
    for sink in sinks:
        G_flow.add_edge(sink, supersink, capacity=1e9)
    
    # Compute maximum flow
    flow_value, flow_dict = nx.maximum_flow(G_flow, supersource, supersink, capacity='capacity', flow_func=nx.algorithms.flow.shortest_augmenting_path)
    return flow_value


def process_sample(sample_udp: pd.Series):
    """Compute pathway activities for one sample using NetworkX."""
    global PATHWAY_GRAPHS
    activities = {}

    for pathway, G in PATHWAY_GRAPHS.items():
        corridors = G.graph.get('corridors', [])        
        if len(corridors) < 2:
            activities[pathway] = shallow_fallback_activity(G, sample_udp)
            continue
        
        # Compute flow for entire pathway at once
        flow_value = compute_pathway_flow(G, corridors, sample_udp)
        activities[pathway] = float(flow_value)
    
    return activities


def parallel_apply(df, pathway_interactions_dict):
    """Applies a function to DataFrame rows in parallel, preserving order."""
    n_cores = max(1, mp.cpu_count() - 2) # leave 2 cores free for OS
    with mp.Pool(n_cores, initializer=init_pathway_graphs, initargs=(pathway_interactions_dict,)) as pool:
        results = list(
            tqdm(
                pool.imap(process_sample, (row for _, row in df.iterrows())),
                total=len(df),
            )
        )
    return pd.DataFrame(results, index=df.index)


PATHWAY_GRAPHS = {}


def init_pathway_graphs(pathway_interactions):
    global PATHWAY_GRAPHS
    for pathway, interactions in pathway_interactions.items():
        PATHWAY_GRAPHS[pathway] = build_pathway_graph_structure(interactions)
    #print(f"Built {len(PATHWAY_GRAPHS)} pathway graphs")


if __name__ == '__main__':
    # Initialize graph structures.
    pathway_interactions = parse_pathway_interactions('./data/pathway_relations.csv')
    init_pathway_graphs(pathway_interactions)
    udp_df = pd.read_csv('./data/TCGACRC_expression-merged.zip', sep='\t', index_col=0)
    udp_df.index = udp_df.index.str.lower()

    DEBUG=False
    if DEBUG:
        results_all = []
        for col in tqdm(udp_df.columns):
            result = process_sample(udp_df[col])
            results_all.append(result)
        results = pd.DataFrame(results_all, index=udp_df.columns).T
    else:
        df_to_process = udp_df.T
        print(f"Processing {len(df_to_process)} samples...")
        results = parallel_apply(df_to_process, pathway_interactions).T
    results = results.round(4)
    results.to_csv('./data/output_activity.csv')
    print(f"Saved results to ./data/output_activity.csv")