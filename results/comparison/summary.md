# Sample-size sweep — summary

## F1 Score

| Model                |    30k |    80k |   150k |   230k |
|:---------------------|-------:|-------:|-------:|-------:|
| Isolation Forest     | 0.3191 | 0.3386 | 0.3031 | 0.2839 |
| Local Outlier Factor | 0.0213 | 0      | 0      | 0      |
| One-Class SVM        | 0.031  | 0.0425 | 0.059  | 0.0709 |
| One-Class SVM (SGD)  | 0      | 0      | 0      | 0      |
| Robust Covariance    | 0      | 0.1575 | 0.0039 | 0.0051 |

## Recall

| Model                |    30k |    80k |   150k |   230k |
|:---------------------|-------:|-------:|-------:|-------:|
| Isolation Forest     | 0.3191 | 0.3386 | 0.3031 | 0.2839 |
| Local Outlier Factor | 0.0213 | 0      | 0      | 0      |
| One-Class SVM        | 0.4894 | 0.5118 | 0.4803 | 0.491  |
| One-Class SVM (SGD)  | 0      | 0      | 0      | 0      |
| Robust Covariance    | 0      | 0.1575 | 0.0039 | 0.0051 |

## Precision

| Model                |    30k |    80k |   150k |   230k |
|:---------------------|-------:|-------:|-------:|-------:|
| Isolation Forest     | 0.3191 | 0.3386 | 0.3031 | 0.2839 |
| Local Outlier Factor | 0.0213 | 0      | 0      | 0      |
| One-Class SVM        | 0.016  | 0.0222 | 0.0314 | 0.0382 |
| One-Class SVM (SGD)  | 0      | 0      | 0      | 0      |
| Robust Covariance    | 0      | 0.1575 | 0.0039 | 0.0051 |

## Total Time (s)

| Model                |    30k |     80k |     150k |     230k |
|:---------------------|-------:|--------:|---------:|---------:|
| Isolation Forest     | 0.2487 |  0.569  |   0.9476 |   1.4324 |
| Local Outlier Factor | 0.6937 |  4.8068 |  17.8764 |  52.2498 |
| One-Class SVM        | 5.8144 | 52.8543 | 136.659  | 357.042  |
| One-Class SVM (SGD)  | 0.0197 |  0.1115 |   0.207  |   0.3354 |
| Robust Covariance    | 2.6033 |  8.5791 |  14.1716 |  19.8959 |

