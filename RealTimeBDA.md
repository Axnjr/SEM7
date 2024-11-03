# 1. PageRank: 
is an algorithm used by Google Search to rank web pages in their search engine results. It measures the importance of web pages based on 
the number and quality of links to them. In PageRank, the web is represented as a graph:
- Each web page is a node.
- Each hyperlink from one page to another is a directed edge. 
Here’s a simplified explanation:

### How PageRank Works
- Basic Idea: Pages that have more links pointing to them are considered more important and are ranked higher.
- Link Weighting: Not all links are equal. Links from highly-ranked pages carry more weight than those from lower-ranked pages.
- Iterative Calculation: PageRank is calculated through an iterative process where the rank of a page is divided among the pages it links to, then summed up to get the PageRank of the linked pages.

### Mathematical Representation
The PageRank of a page `p` is defined as: <br>
`PR(p) = (1 - d / n) + d * ∑ (PR(i) / L(i))`<br>
- `d`: damping factor (usually set to 0.85) to simulate the probability of a random surfer continuing to click on links.
- `n`: Total number of pages.
- `PR(i)`: PageRank of page `i` that links to page `P`
- `L(i)`: Number of outbound links on page `i`.

### Example

Imagine three pages: A, B, and C.

Page A links to B and C.

Page B links to C.

Page C links to A and B.

Initially, each page is given an equal PageRank. Through iterative calculations, PageRank distributes the rank values based on the link structure, resulting in final PageRank values that reflect their importance.

### Applications
- Search Engines: Used by Google to rank web pages in search results.

- Social Networks: Analyzing the importance and influence of individuals within a network.

PageRank revolutionized the way search engines worked by focusing on link structure rather than just content, providing more relevant and reliable search results.


# 2. Working outline of `pageRank`:

- Construct the Graph of Pages:
    - Each webpage is a node.
    - Each hyperlink from one page to another is a directed edge.
- Convert the Graph to a Sparse Matrix:
    - Build an adjacency matrix (link matrix) `𝐿`, where: 
    - `𝐿[𝑖][𝑗] = 1`if page j links to page i; otherwise, it’s 0.
    - Transform `L` into a transition matrix `T` by 
    normalizing each column, so each column sums to 1. 
    This represents the probability of "jumping" to each 
    linked page.
- Construct the PageRank Matrix with Damping:
    - Define the PageRank matrix `𝑀` with damping factor `𝛼`
    - same `pageRank` formula.
- Assign an initial PageRank to each page, typically 1/N, 
where N is the total number of pages. Update PageRank values 
using the formula.

# 3. Efficient computation of PageRank: 
is crucial for handling large-scale web graphs. Here are a few methods to achieve this: 

## 1: Power Iteration Method
This is the most common method for computing PageRank. It involves iteratively updating the PageRank values until convergence. 
Initialization: Assign an initial PageRank to each page, typically `1/N`, where `N` is the total number of pages.
Iteration: Update PageRank values using the formula:
![page rank formula](image-8.png)

## 2: The sparse matrix method: 
is an efficient way to store and manipulate matrices that contain a large number of zero elements. In a sparse matrix, only the 
non-zero elements are stored along with their row and column indices. This reduces memory usage significantly compared to storing 
all elements, including zeros. This methods is particularly usefull in Calculating `pageRank` of pages in serach engines, as  
most pages only link to a small fraction of other pages in the real-wrold scenario.

### Storage Formats:
- Compressed Sparse Row (CSR): Stores non-zero elements along with the row pointers and column indices. This format is efficient for row slicing and matrix-vector multiplication.
- Compressed Sparse Column (CSC): Similar to CSR but optimized for column slicing operations.
- Coordinate List (COO): Stores a list of (row, column, value) tuples. It's easy to construct but less efficient for arithmetic operations compared to CSR and CSC.

### Advantages
- Memory Efficiency: Only non-zero elements are stored, reducing memory usage.
- Computational Efficiency: Sparse matrix operations are faster due to reduced data access and manipulation.

### Applications
- Search Engines: Efficiently computing PageRank for large web graphs.
- Graph Algorithms: Handling large-scale networks in social media, transportation, and biological networks.

## 3: Parallel and Distributed Computing
For very large graphs, parallel and distributed computing techniques are employed.
- MapReduce: Implement the PageRank algorithm using the MapReduce programming model to distribute the computations across multiple nodes.
- Graph Processing Systems: Use specialized graph processing frameworks like Apache Giraph or Pregel that are optimized for handling large-scale graph computations.

# 4. PageRank using MapReduce:
MapReduce is a programming model for processing large-scale data across a distributed computing environment. It consists 
of two main phases:
- Map Phase: Processes input data and produces intermediate key-value pairs.
- Reduce Phase: Merges intermediate values associated with the same key.

### Step-by-Step Process

- Mapper Phase: For each page `𝑃` with current PageRank 
`𝑃𝑅(𝑃)` and a list of `outbound links out(p)`, the mapper:
Emits contributions to each linked page based on `PR(p) / out(p)`. 
Example output from a mapper for page 
`A` linking to pages `𝐵`, and `𝐶`:
    - Input: (Page A, [Page B, Page C])
    - Output: (Page B, PR(A) / 2), (Page C, PR(A) / 2)

- Reducer Phase:
Sum the PageRank contributions from all inbound links to a page.
Apply the damping factor `d` and adjust for random jump factor `1 - d / n`
Example:
    - Input: (Page B, [PR(A)/2, PR(C)/3])
    - Output: PR(B) = `(1 - d / N) + d (PR(A)/2 + PR(C)/3)`

- Iteration:
Repeat the Map and Reduce steps for a fixed number of iterations or until the PageRank values converges.

- Pseudo-code for PageRank using MapReduce:
    ```py
    # Map Function
    def map(url, links, rank):
        for link in links:
            yield (link, rank / len(links))
        yield (url, 0)

    # Reduce Function
    def reduce(url, ranks):
        new_rank = (1 - d) / N + d * sum(ranks)
        return new_rank
    ```
### Benefits of Using MapReduce
- Scalability: Can handle large datasets by distributing the computation across multiple machines.
- Efficiency: Parallel processing reduces computation time significantly.
- Fault Tolerance: Built-in fault tolerance mechanisms handle failures in a distributed environment.

By leveraging MapReduce, computing PageRank for a massive web graph becomes feasible and efficient, ensuring that the ranking algorithm can keep up with the ever-growing size of the internet.