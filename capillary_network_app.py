import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_network(G):
    # Créer un graphe NetworkX
    graph = nx.DiGraph()

    # Ajouter les nœuds
    nb_nodes = len(G)
    graph.add_nodes_from(range(nb_nodes))

    # Ajouter les arêtes avec poids
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if G[i, j] != 0:
                graph.add_edge(i, j, weight=G[i, j])

    # Position des nœuds (spring layout)
    pos = nx.spring_layout(graph)

    # Dessiner le graphe
    plt.figure(figsize=(8, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=200,
        edge_color="gray",
        font_size=12,
        font_weight="bold",
        arrowsize=20,
    )

    # Ajouter les poids des arêtes
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    # Retourner l'objet Matplotlib
    return plt

st.title("Flough through a capillary network")

G_input = st.text_area(
    "Enter the adjancency matrix \(G\) (format : [[0, 0.3, 0], [0.3, 0, 0.4], ...]):",
    ""
)
try:
    G = np.array(eval(G_input))

    if G.shape[0] != G.shape[1]:
        st.error("The matrix \(G\) must be square.")
    elif not np.allclose(G, G.T):
        st.error("The matrix \(G\) must be symetrical.")
    else:
        st.success("The matrix \(G\) is valid.")

        if st.checkbox("Plot the capillary network"):
            fig = plot_network(G)
            st.pyplot(fig)

        pa = st.number_input("Arterial pressure (Pa)", value=0.0, step=1.0)
        pv = st.number_input("Venous pressure (Pv)", value=0.0, step=1.0)
        mu = st.number_input("Dynamic viscosity (µ) in Pa.s", value=0, format="%.6f")
        r = st.number_input("Capillary radius (r) in meters", value=0, format="%.9f")

        if st.button("Compute the blood flow"):
            R = 8 * mu / (np.pi * r**4)

            nb_nodes = len(G)
            A = np.zeros([nb_nodes, nb_nodes])
            A[0, 0] = 1
            A[nb_nodes - 1, nb_nodes - 1] = 1

            b = np.zeros(nb_nodes)
            b[0] = pa
            b[nb_nodes - 1] = pv

            for i in range(1, nb_nodes - 1):
                for j in range(nb_nodes):
                    if G[i, j] != 0:
                        A[i, j] = -1 / G[i, j]
                        A[i, i] += 1 / G[i, j]

            try:
                p = np.linalg.solve(A, b)
                st.write("Pressure at the nodes:")
                st.dataframe(pd.DataFrame(p, columns=["Pressure (mmHg)"], index=[f"Node {i+1}" for i in range(len(p))]))

                Q = np.zeros([nb_nodes, nb_nodes])
                Q_simp = []
                for i in range(nb_nodes):
                    for j in range(nb_nodes):
                        if G[i, j] != 0:
                            Q[i, j] = 133.32 * np.abs(p[i] - p[j]) / (R * G[i, j] / 1000)
                            if j>i: 
                                Q_simp.append(Q[i,j])

                # Q_simp = np.round(np.array(Q_simp), decimals=18)
                formatted_Q_simp = [f"{q:.5e}" for q in Q_simp]
                st.write("Flow in the capillaries (m³/s) :")
                st.dataframe(pd.DataFrame(np.array(formatted_Q_simp), columns=["Flow (m³/s)"], index=[f"Branch {i+1}" for i in range(len(p)+1)]))

            except np.linalg.LinAlgError:
                st.error("Error in solving equations. Check matrix \(G\).")
except:
    st.error("Please enter a valid matrix (square and symetrical).")