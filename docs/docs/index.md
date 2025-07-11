# mloda 

## Transforming Data and Feature Engineering

mloda **rethinks data and feature engineering** by offering a **flexible, resilient framework** that adapts seamlessly to changes. It focuses on defining transformations rather than static states, facilitating smooth transitions between development phases, and reducing redundant work. 

Teams can efficiently develop MVPs, scale to production, and adapt systems—all while maintaining high data quality, governance, and scalability.
[Get started with mloda can be found here.](https://mloda-ai.github.io/mloda/chapter1/installation/)

mloda's plug-in system **automatically selects the right plugins for each task**, enabling efficient querying and processing of complex features. [Learn more about the mloda API here.](https://mloda-ai.github.io/mloda/in_depth/mloda-api/) By defining feature dependencies, transformations, and metadata processes, mloda minimizes duplication and fosters reusability.

mloda's framework also allows **plug-ins to be shared and reused through a centralized repository**. This ensures consistency, reduces operational complexity, and promotes best practices. This collaborative approach significantly reduces redundant work.

Finally, mloda aims to **unify various tools** such as data catalogs, feature stores, versioning systems, metadata stores, data contracts, and data lineage tools into one cohesive solution.

## Key Benefits 

The benefits are not limited to the features listed below.

**Feature Engineering and Data Processing**

- automated feature engineering
- data cleaning
- synthetic data generation
- time travel

**Data Management and Ownership**

- one data source
- rich metadata like data lineage, data usage
- clear split roles by users, engineers and owners speaking same language

**Data Quality and Security**

- data quality definitions
- unit- and integration tests
- secure queries

**Scalability**

- switch compute framework without changing feature logic
- multi-environment support (offline, online, migrations)

**Community Engagement by Design**

- shareable plug-in ecosystem
- fostering community

## Core Components and Architecture

mloda addresses common challenges in data and feature engineering by leveraging two key components:

#### Plugins
  - Feature Groups: **Define feature dependencies**, such as creating a composite label based on features e.g. user activity, purchase history, and support interactions. Once defined, only the label needs to be requested, as dependencies are resolved automatically, simplifying processing. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)

  - Compute Frameworks: Defines the **technology stack**, like Spark or Pandas, along with support for different storage engines such as Parquet, Delta Lake, or PostgreSQL, to execute feature transformations and computations, ensuring efficient processing at scale. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/compute-frameworks/)

  - Extenders: Automates **metadata extraction processes**, helping you enhance data governance, compliance, and traceability, such as analyzing how often features are used by models or analysts, or understanding where the data is coming from. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/extender/)

#### Core
  - Core Engine: **Handles dependencies between features and computations** by coordinating linking, joining, filtering, and ordering operations to ensure optimized data processing. For example, in customer segmentation, the core engine would link and filter different data sources, such as demographics, purchasing history, and online behavior, to create relevant features.

## Contributing to mloda

-   We welcome contributions from the community to help us improve and expand mloda. Whether you're interested in developing plug-ins, or adding new features, your input is invaluable. [Learn more here.](https://mloda-ai.github.io/mloda/development/)


## Frequently Asked Questions (FAQ)

If you have additional questions about mloda and how it can enhance your data and feature engineering workflow visit our [FAQ](https://mloda-ai.github.io/mloda/faq) section, raise an [issue](https://github.com/mloda-ai/mloda/issues/) on our GitHub repository, or email us at [info@mloda.ai](mailto:info@mloda.ai).


## License

This project is licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE.TXT).
