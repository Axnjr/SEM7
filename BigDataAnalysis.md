# Questions:

What is big data? challenges of big data.  

Traditional systems and big data approach. 

Five V’s of Big Data. 

Types of big data with examples. 

Define Hadoop and its limitations. 

Explain how node failure is handled in Hadoop. 

Job tracker and task tracker. 

Hadoop core components with its Ecosystem. 

CAP theorem. 

HDFS architecture in detail with diagram. 

Explain MapReduce execution pipeline with suitable example.  

Write a MapReduce pseudo code to multiply two matrices. Apply map reduce working to perform given matrix multiplication. 

Illustrate MapReduce to perform the relational algebraic operations of grouping and aggregations/Union on the given dataset? 

List transaction properties of NoSQL. 

 Business drivers behind the NoSQL. 

 Key-value pair database architecture pattern. 

How do the strategies employed by NoSQL databases solve the challenges associated with    managing big data? 

NoSQL case studies. 

Shared nothing architecture in detail. 

Explain distributing models in details. 

Explain how NoSQL systems handle the big data problems. 

<br>

# 1. Big data: 
refers to extremely large datasets that are too complex to be processed and analyzed using traditional data processing tools. These datasets are characterized by the “5 V’s”:
- Volume: 
The sheer amount of data generated.
- Velocity: 
The speed at which new data is generated and processed.
- Variety: 
The different types of data (structured, unstructured, semi-structured).
- Veracity: 
The quality and accuracy of the data. This involves ensuring the data is trustworthy and reliable.
- Value: 
The potential insights and benefits that can be derived from analyzing the data.

# 2 . Challenges of Big Data
- Data Quality: 
Ensuring the accuracy, completeness, and reliability of data.
- Data Integration: 
Combining data from different sources and formats.
- Storage and Management: 
Efficiently storing and managing vast amounts of data.
- Data Security: 
Protecting sensitive data from breaches and unauthorized access.
- Scalability: 
Ensuring systems can handle increasing amounts of data.
- Data Analysis: 
Extracting meaningful insights from large datasets.
- Cost: 
Managing the expenses associated with big data infrastructure and tools

# 3. Traditional Systems vs. Big Data Approach
> Traditional Systems

- Data Volume: 
Designed to handle smaller datasets, typically in gigabytes or terabytes.
- Data Processing: 
Relies on batch processing, where data is collected, processed, and then analyzed.
- Data Storage: 
Uses relational databases (RDBMS) with structured data and predefined schemas.
- Scalability: 
Limited scalability; scaling up often requires more powerful hardware.
- Data Variety: 
Primarily handles structured data.
- Data Analysis: 
Uses traditional data analysis tools and techniques.

> Big Data Approach

- Data Volume: 
Capable of handling massive datasets, often in petabytes or exabytes.
- Data Processing: 
Utilizes both batch and real-time processing to handle continuous data streams.
- Data Storage: 
Employs distributed storage systems like Hadoop HDFS, NoSQL databases, and cloud storage to manage diverse data types.
- Scalability: 
Highly scalable; can scale out by adding more nodes to the system.
- Data Variety: 
Manages structured, unstructured, and semi-structured data.
- Data Analysis: 
Leverages advanced analytics, machine learning, and AI to extract insights from large datasets.

> Key Differences
- Scalability: 
Big data systems are designed to scale horizontally, adding more servers to handle increased load, whereas traditional systems often scale vertically, requiring more powerful hardware.
- Data Types: 
Big data systems can handle a wider variety of data types, including text, images, videos, and sensor data, while traditional systems are more limited to structured data.
- Processing Speed: 
Big data systems can process data in real-time, providing faster insights, whereas traditional systems often rely on batch processing, which can be slower.

# 4. Big data Types: 
> can be categorized into several types based on its structure and source. Here are the main types along with examples:

1. Structured Data
Definition: Data that is organized in a fixed format, often in rows and columns.
Examples:
Databases: Customer information in a CRM system.
Spreadsheets: Sales data in Excel sheets.

2. Unstructured Data
Definition: Data that does not have a predefined format or structure.
Examples:
Text Files: Emails, social media posts, and documents.
Multimedia: Images, videos, and audio files.

3. Semi-Structured Data
Definition: Data that does not conform to a rigid structure but has some organizational properties.
Examples:
XML/JSON Files: Data exchanged between web services.
Log Files: Server logs and event logs.

4. Geospatial Data
Definition: Data that includes geographical components.
Examples:
Maps: GPS data, satellite imagery.
Location Data: Data from mobile devices indicating user locations.

5. Machine-Generated Data
Definition: Data created by machines without human intervention.
Examples:
Sensor Data: Data from IoT devices, such as temperature sensors.
Web Logs: Data generated by web servers tracking user activity.

6. Human-Generated Data
Definition: Data created by humans through various activities.
Examples:
Social Media: Tweets, Facebook posts, and Instagram photos.
Documents: Word documents, PDFs, and presentations.

# 5. Hadoop: 
is an open-source software framework used for distributed storage and processing of large datasets. It leverages the power of distributed computing to handle massive amounts of data across clusters of computers. The core components of Hadoop include:

- Hadoop Distributed File System (HDFS): 
A distributed file system that stores data across multiple machines.
- MapReduce: 
A programming model for processing large datasets with a parallel, distributed algorithm.
- YARN (Yet Another Resource Negotiator): 
Manages and schedules resources in the cluster.
- Hadoop Common: 
Provides common utilities and libraries that support the other Hadoop modules
- Ecosystem Tools
    - Hive: 
    Data warehousing and SQL-like query language.
    - Pig: 
    High-level platform for creating MapReduce programs.
    - HBase: 
    NoSQL database for real-time read/write access to large datasets.
    - Spark: 
    In-memory data processing engine.
    - Mahout: 
    Machine learning library.
    - Sqoop: 
    Tool for transferring data between Hadoop and relational databases.
    - Flume: 
    Service for collecting and moving large amounts of log data.
    - Oozie: 
    Workflow scheduler to manage Hadoop jobs.
    - Zookeeper: 
    Coordination service for distributed applications

# 6. Limitations of Hadoop
> Despite its advantages, Hadoop has several limitations:

- Small File Handling: 
Hadoop is inefficient at handling a large number of small files, as HDFS is optimized for large files.
- Real-Time Processing: 
Hadoop is designed for batch processing and is not suitable for real-time data processing.
- Iterative Processing: 
It is not efficient for iterative processing tasks, which are common in machine learning algorithms.
- Latency: 
The MapReduce framework can introduce significant latency, making it less suitable for low-latency applications.
- Complexity: 
Setting up and managing a Hadoop cluster can be complex and requires specialized knowledge.
- Security: 
Hadoop’s security model is not as robust as some other systems, which can be a concern for sensitive data.
- Resource Management: 
While YARN improves resource management, it can still be challenging to optimize resource allocation in large clusters.

> These limitations have led to the development of other frameworks like Apache Spark and Apache Flink, which address some of these issues.

# 7. Node failures in Hadoop
In Hadoop, node failure is managed through several mechanisms to ensure data reliability and availability. Here’s how it works:

> Handling DataNode Failure

- Heartbeat Mechanism: 
Each DataNode sends a heartbeat signal to the NameNode at regular intervals (typically every 3 seconds). This signal indicates that the DataNode is functioning properly.

- Block Reports: 
DataNodes also send block reports to the NameNode, detailing all the blocks they store.
> Blocks: files are split into blocks (default size is 128 MB) and stored across DataNodes.

- Detection of Failure: 
If the NameNode does not receive a heartbeat from a DataNode for a specified period (usually 10 minutes), it marks the DataNode as dead.

- Replication: 
Hadoop’s HDFS is designed to replicate data blocks across multiple DataNodes. When a DataNode fails, the NameNode ensures that the blocks stored on the failed DataNode are replicated to other DataNodes to maintain the desired replication factor.

- Rebalancing: 
The NameNode may also initiate a rebalancing process to distribute the data evenly across the remaining DataNodes.

> Handling NameNode Failure

- High Availability (HA): 
In a high-availability setup, Hadoop can have multiple NameNodes (active and standby). If the active NameNode fails, the standby NameNode takes over to ensure continuous operation.

- Checkpointing: 
The NameNode periodically saves the namespace and transaction logs to disk. This process, known as checkpointing, helps in recovering the state of the NameNode in case of a failure.

>Handling Task Failure in MapReduce

- Task Retries: 
If a task fails, the Application Master retries the task on a different node. The number of retries can be configured.

- Speculative Execution: 
Hadoop can run multiple instances of the same task on different nodes. The first instance to complete successfully is used, and the others are killed. This helps in mitigating the impact of slow or failing nodes.

These mechanisms ensure that Hadoop can handle node failures gracefully, maintaining data integrity and availability.

# 8. HDFS Architecture
Hadoop Distributed File System (HDFS) is designed to store large datasets reliably and to stream those data sets at high bandwidth to user applications. Here’s a detailed look at its architecture: Key Components

> NameNode
- Role: 
Acts as the master server that manages the file system namespace and regulates access to files by clients.
- Functions:
Maintains the file system tree and the metadata for all the files and directories. Keeps track of the DataNodes where the actual data blocks are stored. Handles the namespace operations like opening, closing, and renaming files and directories.

> DataNode

- Role: 
Acts as the worker nodes that store the actual data.
- Functions:
Responsible for serving read and write requests from the file system’s clients. Performs block creation, deletion, and replication upon instruction from the NameNode. Sends regular heartbeat signals to the NameNode to report its status and the status of the blocks it stores.

> Secondary NameNode
- Role: 
Assists the primary NameNode.
- Functions:
Periodically merges the namespace image with the edit log to prevent the edit log from becoming too large. Acts as a checkpoint node but does not serve as a backup NameNode.
Checkpoint Node and Backup Node

> Checkpoint Node: 
Periodically creates checkpoints of the namespace.

> Backup Node: 
Provides a read-only copy of the file system metadata.
Data Storage and Replication

- Blocks: 
Files are split into blocks (default size is 128 MB) and stored across DataNodes.
- Replication: 
Each block is replicated across multiple DataNodes (default replication factor is 3) to ensure fault tolerance.

> Data Read and Write Operations
- Write Operation:
    - The client contacts the NameNode to create a new file.
    - The NameNode checks for file existence and permissions.
    - The client writes data to the DataNodes in a pipeline fashion.
    - The DataNodes replicate the data blocks to other DataNodes.
- Read Operation:
    - The client contacts the NameNode to get the locations of the data blocks.
    - The client reads the data directly from the DataNodes.
    ![hdfs](image-1.png)

# 9. The CAP theorem: 
also known as Brewer’s theorem, is a fundamental principle in distributed systems, particularly relevant to big data. It states that a distributed data store can only provide two out of the following three guarantees simultaneously:

- Consistency: 
means that all clients see the same data at the same time, no matter which node they connect to. For this to happen, whenever data is written to one node, it must be instantly forwarded or replicated to all the other nodes in the system before the write is deemed ‘successful.’

- Availability: 
means that any client making a request for data gets a response, even if one or more nodes are down. Another way to state this—all working nodes in the distributed system return a valid response for any request, without exception.

- Partition tolerance
A partition is a communications break within a distributed system—a lost or temporarily delayed connection between two nodes. Partition tolerance means that the cluster must continue to work despite any number of communication breakdowns between nodes in the system.

> In the context of big data, the CAP theorem helps in designing and understanding the trade-offs in distributed systems. For example, NoSQL databases often prioritize availability and partition tolerance over consistency to handle large volumes of data and ensure system reliability.

