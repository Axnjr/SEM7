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
