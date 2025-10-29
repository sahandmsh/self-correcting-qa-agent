from datasets import load_dataset

class DataSetUtil:
    def __init__(self):
        pass


    def load_qa_dataset(self, dataset_name: str = "squad", samples_count: int = 5000, random_sampling: bool = False, dataset_split: str = "train"):
        """ Loads QA dataset and returns expected number of samples
        Args:
            samples_count (int): number of samples
            random_sampling (bool): return random samples if True
            dataset_split (str): select with part of dataset to use (e.g., train, test)
        returns:
            list: a list of sampled items
        """
        dataset = load_dataset(dataset_name, split = dataset_split)
        if random_sampling:
            dataset = dataset.shuffle()
        dataset = dataset.select(range(min(samples_count, len(dataset))))
        sampled_data = [
            {'id': item['id'],
            'question': item['question'],
            'context':item['context'],
            'answers':item['answers']} for item in dataset
        ]
        return sampled_data