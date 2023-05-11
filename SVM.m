% Start the stopwatch timer
tic;

% Import the data
data = readtable('IMDB Dataset.csv');

% Preprocess the data
data.review = lower(data.review);
data.review = regexprep(data.review, '<[^>]+>', '');
data.review = regexprep(data.review, '[^a-zA-Z\s]', '');
data.sentiment = categorical(data.sentiment);

% Tokenize the reviews
data.review = tokenizedDocument(data.review);

% Split the data into training and testing sets
cv = cvpartition(data.sentiment, 'HoldOut', 0.2);
train_data = data(cv.training,:);
test_data = data(cv.test,:);

% Create a bag-of-words model 
bag = bagOfWords(train_data.review);
X_train = encode(bag, train_data.review);
X_test = encode(bag, test_data.review);
y_train = train_data.sentiment;
y_test = test_data.sentiment;

% Train the linear classification model
linearModel = fitclinear(X_train, y_train, 'Learner', 'svm');

% Make predictions
y_pred = predict(linearModel, X_test);

% Calculate evaluation metrics
confusionMatrix = confusionmat(y_test, y_pred);
TP = confusionMatrix(1,1);
TN = confusionMatrix(2,2);
FP = confusionMatrix(1,2);
FN = confusionMatrix(2,1);

% Stop the stopwatch timer
elapsedTime = toc;

Precisionsvm = TP / (TP + FP);
recallsvm = TP / (TP + FN);
f1_scoresvm = 2 * (Precisionsvm * recallsvm) / (Precisionsvm + recallsvm);
accuracysvm = mean(y_pred == y_test);

fprintf('Total running time: %s.\n', datestr(datenum(0,0,0,0,0,elapsedTime),'HH:MM:SS'));
fprintf('Accuracy: %.2f%%\n', accuracysvm * 100);
fprintf('Precision: %.2f%%\n', Precisionsvm * 100);
fprintf('Recall: %.2f%%\n', recallsvm * 100);
fprintf('F1-score: %.2f%%\n', f1_scoresvm * 100);
