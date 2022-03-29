from pyspark import SparkContext
import sys, time, csv, json
import xgboost as xgb


def print_line(x, item):
    return tuple((item[0], item[1], x))


def print_file(result, test_list):
    with open(OUTPUT_FILE, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(('user_id', ' business_id', ' prediction'))
        for x, y in zip(result, test_list):
            writer.writerow(print_line(x, y))


def get_user_json_data(file_path):
    user_rdd = spark_context.textFile(file_path) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], (x['average_stars'], x['review_count'], x['fans'])))

    return user_rdd.collectAsMap()


def get_business_json_data(file_path):
    business_rdd = spark_context.textFile(file_path) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))

    return business_rdd.collectAsMap()


def get_review_train_data(file_path):
    return file_path.map(lambda x: [user_data_as_map[x['user_id']][0], user_data_as_map[x['user_id']][1], user_data_as_map[x['user_id']][2], business_data_as_map[x['business_id']][0], business_data_as_map[x['business_id']][1]])


def get_labels_data(rdd):
    return rdd.map(lambda x: x['stars'])


def get_mapping(item):
    feat1 = user_data_as_map[item[0]][0]
    feat2 = user_data_as_map[item[0]][1]
    feat3 = user_data_as_map[item[0]][2]
    feat4 = business_data_as_map[item[1]][0]
    feat5 = business_data_as_map[item[1]][1]

    return [feat1, feat2, feat3, feat4, feat5]


def get_test_data(rdd):
    return rdd.map(lambda x: get_mapping(x))


def test_data_map(item):
    a = item.split(',')[0]
    b = item.split(',')[1]
    return (a, b)


if __name__ == '__main__':
    start = time.time()

    INPUT_TRAIN_FOLDER = sys.argv[1]
    INPUT_TEST_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    spark_context = SparkContext().getOrCreate()
    spark_context.setLogLevel("ERROR")

    # Reading from the input file user.json
    user_data_as_map = get_user_json_data(INPUT_TRAIN_FOLDER + '/user.json')

    # Reading from the input file business.json
    business_data_as_map = get_business_json_data(INPUT_TRAIN_FOLDER + '/business.json')

    # Reading from the input file review_train.json
    review_rdd = spark_context.textFile(INPUT_TRAIN_FOLDER + '/review_train.json').map(lambda x: json.loads(x))
    train_rdd = get_review_train_data(review_rdd)
    labels_rdd = get_labels_data(review_rdd)

    # X = get_X(train_rdd)
    # y = get_y(labels_rdd)
    X = train_rdd.collect()
    y = labels_rdd.collect()

    model = xgb.XGBRegressor()
    model.fit(X, y)

    test_dataset_rdd = spark_context.textFile(INPUT_TEST_FILE)\
        .map(lambda x: test_data_map(x))

    header = test_dataset_rdd.first()
    test_dataset_rdd = test_dataset_rdd.filter(lambda x: x != header)
    test_data_result = test_dataset_rdd.collect()

    test_data_rdd = get_test_data(test_dataset_rdd)

    test_X = test_data_rdd.collect()
    result = model.predict(test_X)

    print_file(result, test_data_result)

    end = time.time()

    print("Duration: ", end - start)
