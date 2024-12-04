import pyterrier as pt
import pandas as pd
import settings as s
# Load an existing index (change the path to your index location)

# Initialize PyTerrier
if not pt.java.started():
    pt.java.init()

index = pt.IndexFactory.of(str(s.dataset_index_dir))

# Access the MetaIndex
meta_index = index.getMetaIndex()

# List the fields available in the MetaIndex
fields = meta_index.getKeyNames()

# Show the available fields
print("Available fields in the MetaIndex:", fields)



def get_details():
    # Example: Getting document metadata
    meta = []
    for docid in range(index.getCollectionStatistics().getNumberOfDocuments()):
        doc_meta = index.getMetaIndex().getAllItems(docid)
        meta.append(doc_meta)

    # Convert to DataFrame
    df = pd.DataFrame(meta, columns=["docid", "metadata_field1", "metadata_field2", ...]) # adjust columns as per your index structure

    # Show DataFrame
    print(df)
