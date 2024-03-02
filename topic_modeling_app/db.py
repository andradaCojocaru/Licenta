from pymongo import MongoClient

def save_to_mongodb(selected_parameters, corpus_name, model_id, model_name):
    # Use Djongo's database connection
    client = MongoClient('mongodb+srv://andradacojocaru:andrada@cluster0.rpknlzf.mongodb.net/')  # Replace 'connection_string' with your actual connection string
    db = client['topic_modelling']  # Replace 'db_name' with your actual database name
    # Choose or create a collection in your database
    collection = db['combined_topic_model']  # Replace 'selected_parameters_collection' with your actual collection name

    existing_data = collection.find_one({
        'corpus_data.corpus_name': corpus_name,
        'selected_parameters': selected_parameters,
        'model_name': model_name
    })

    if existing_data:
        print("Data with the same corpus name, model name, and parameters already exists.")
        return existing_data['model_id'], True, True
    
    # Check if data with the same corpus name exists
    existing_corpus_data = collection.find_one({
        'corpus_data.corpus_name': corpus_name
    })
    text_id = model_id

    if existing_corpus_data:
        print("Data with the same corpus name already exists, but parameters or model name may differ.")
        text_id = existing_corpus_data['text_id']
    
    combined_data = {
        'selected_parameters': selected_parameters,
        'corpus_data': {
            'corpus_name': corpus_name
        },
        'model_id': model_id,
        'text_id': text_id,
        'model_name': model_name 
    }
    # Insert the selected parameters into the collection
    collection.insert_one(combined_data)
    if existing_corpus_data:
        return text_id, False, True
    return model_id, False, False