# Sample Hosted fastText Service

This sample implements two [`mlservicewrapper-core`](https://github.com/ml-service-wrapper/ml-service-wrapper-core) services to expose [fastText](https://fasttext.cc/docs/en/language-identification.html) functionality:

- Language identification
- Vectorization

# Install and Run

## 0. Prerequisites

This tutorial assumes you have:

- A git client installed
- Python 3.6+ installed

## 1. Clone this repository

```bash
git clone https://github.com/ml-service-wrapper/ml-service-wrapper-sample-fasttext.git
cd ml-service-wrapper-sample-fasttext
```

### 1a. (Optional) Create and activate a virtual environment

```bash
virtualenv venv
source ./venv/bin/activate
```

## 2. Install dependencies

### 2a. `mlservicewrapper-core`

This includes all core modules you'll need to implement a Service, including a basic debug host.

```bash
pip install mlservicewrapper-core
```

## 2b. Install extra dependencies

You'll need a copy of `fastText`, which can occassionally have complexities during installation. See their [Get Started](https://fasttext.cc/docs/en/support.html#building-fasttext-python-module) documentation for details and instructions, and be sure to build and install the Python module.

You'll also need `urllib3`, which can be installed using pip:

```bash
pip install urllib3
```

## 3. Run the debug module

With the source code local, everything should work. Simply run the debug module, passing in the appropriate configuration file and sample file as arguments:

```bash
python -m mlservicewrapper.core.debug \
    --config "./config/language_detection.json" \
    --load-params ModelPath=./models/langdetect.bin \
    --input-paths Data=./data/input/multiple_languages.csv \
    --output-dir "./data/output"
```

## 4. Review results

Open the generated file at `./data/output/Results.csv` to see detected languages.

# What's going on?

The flow of the debug execution is easily traceable:

1. The file at the `--config` argument is parsed. In our case, it only has two parameters:

   ```json
   {
     "modulePath": "../src/service.py",
     "className": "LanguageDetectionService"
   }
   ```

   The debug module looks for the module at `modulePath`, _relative to the location of the configuration file._ This helps improve portability of the code-base, especially for hosting in production.

   Once our `service.py` script is imported, the debug module looks for a class defined in it that matches the `className` property, `LanguageDetectionService`.

   The debug creates an instance of that service, builds out a context (more on that in a moment), then calls `load` on it.

2. Because `LanguageDetectionService` inherits from `FastTextServiceBase`, it's really the implementation of `FastTextServiceBase` that gets called. It uses methods to allow different implementations, but flattened out, its first step is to read the `ModelPath` parameter. Because we're running this in debug, the main source for this value is the `--load-params` argument. Since we defined `./models/langdetect.bin`, it will check whether that file exists.

   If this is your first time running the sample (or if you've deleted that file), the `FastTextServiceBase` implementation will call its own `get_model_url` function. Because fastText has a pre-built model for language identification, we just use that. Notice, however, that the `FastTextVectorizerService`, outside scope of this walkthrough, would look for a parameter `ModelUrl`.

   With that url in-hand, the file is downloaded to the location at `ModelPath`. This step essentially primes the cache for future executions.

   Finally, knowing it has a model accessible, it calls `fasttext.load_model`, and stores the result to `self.model` for use later.

3. Being a debug run, the debug module moves right along into the next phase. In production environments, depending on the host, this may happen immediately, or it could be a while (e.g. if exposing an HTTP API).

4. Back in `LanguageDetectionService`, `process` gets called. This is where all of our data-specific prediction logic lives, and it has several steps.

   First, it asks the given `ProcessContext` for a dataframe called `Data`. Hosting environments are left to figure out where to source a result, but in our run, it will first check the `--input-paths` argument. Since we mapped `Data` to `./data/input/multiple_languages.csv`, the context will simply load that file into a dataframe and return it.

   With that dataframe available, a quick validation check is run: if the dataframe is missing a `Text` column, we want to raise a descriptive error, hence `mlservicewrapper.core.errors.MissingDatasetFieldError`. Fortunately our data file _does_ have that column, and we're free of this error.

   A similar test is also run to ensure a field called `Id` is available. The `Id` field isn't used for prediction, but gets echoed back in the result dataframe as a useful piece of context for callers.

   Now's when we start real ML logic which would vary _heavily_ depending on model and desired outcome. For our implementation, the first step is to lightly clean the input text field, then call `DataFrame.apply` to build a prediction dataframe for the results, add column labels, and finally re-insert the `Id` field.

   With the result dataframe all ready, we make a final call to our `ProcessContext.set_output_dataframe`, naming the output `Results` and passing them to be returned.

   Similarly to the input dataframe, it's up to the host environment to decide what to do with the dataframe. The debug module sees our `--output-dir` argument, and saves the contents of the dataframe to a csv in that directory named after the dataset itself: specifically, `Results.csv`.

# (Optional) Run as an HTTP service

Now that the debug service is able to run with the debug module, it's trivial to deploy it as a HTTP service using the [`mlservicewrapper.host.http` module.](https://github.com/ml-service-wrapper/ml-service-wrapper-host-http). Simply install the package and run it.

```bash
pip install mlservicewrapper-host-http

python -m mlservicewrapper.host.http --config ./config/language_detection.json --prod
```

In another window, you can use cURL to make calls against the hosted API.

```bash
curl --header "Content-Type: application/json" \
    --request POST \
    --data '{
    "inputs": {
        "Data": [
            { "Id": 1, "Text": "This is a test" },
            { "Id": 2, "Text": "Dies ist ein Test" }
        ]
    }
}' \
    http://localhost:5000/api/process/batch
```

You should get a response back that looks like this:

```json
{
  "outputs": {
    "Results": [
      { "Id": 1, "Label": "en", "Score": 0.967 },
      { "Id": 2, "Label": "de", "Score": 1.000 }
    ]
  }
}
```
