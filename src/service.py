import json
import os
import shutil

import fasttext
import mlservicewrapper
import mlservicewrapper.core.contexts
import mlservicewrapper.core.errors
import mlservicewrapper.core.services
import pandas as pd
import urllib3

class FastTextServiceBase(mlservicewrapper.core.services.Service):

    async def load(self, ctx: mlservicewrapper.core.contexts.ServiceContext):
        model_path = self.get_model_path(ctx)

        if not os.path.exists(model_path):
            model_parent_path = model_path.rsplit(os.sep, 1)[0]
            if not os.path.exists(model_parent_path):
                os.makedirs(model_parent_path)
            url = self.get_model_url(ctx)
            self.download_model(url, model_path)

        self.model = fasttext.load_model(model_path)

    @staticmethod
    def download_model(model_url: str, model_path: str):
        http = urllib3.PoolManager()
        r = http.request('GET', model_url, preload_content=False)

        with open(model_path, 'wb') as out:
            shutil.copyfileobj(r, out)

        r.release_conn()

    def get_model_path(self, ctx: mlservicewrapper.core.contexts.ServiceContext):
        return ctx.get_parameter_value("ModelPath", required=True)
    
    def get_model_url(self, ctx: mlservicewrapper.core.contexts.ServiceContext):
        return ctx.get_parameter_value("ModelUrl", required=True)

class FastTextVectorizerService(FastTextServiceBase):

    async def process(self, ctx: mlservicewrapper.core.contexts.ProcessContext):
        input_data = await ctx.get_input_dataframe("Data")

        if "Text" not in input_data.columns:
            raise mlservicewrapper.core.errors.MissingDatasetFieldError("Data", "Text")

        input_data["Vector"] = input_data["Text"].str.replace("\n", "").apply(lambda x: json.dumps(self.model.get_sentence_vector(x).tolist()))
        input_data.drop(["Text"], inplace=True, axis="columns")

        await ctx.set_output_dataframe("Results", input_data)

class LanguageDetectionService(FastTextServiceBase):

    def predict(self, text: str):
        t = self.model.predict(text)

        label: str = t[0][0]
        score: float = t[1][0]

        label = label.replace("__label__", "")

        return (label, score)

    async def process(self, ctx: mlservicewrapper.core.contexts.ProcessContext):
        input_data = await ctx.get_input_dataframe("Data")

        if "Text" not in input_data.columns:
            raise mlservicewrapper.core.errors.MissingDatasetFieldError("Data", "Text")

        if "Id" not in input_data.columns:
            raise mlservicewrapper.core.errors.MissingDatasetFieldError("Data", "Id")

        cleaned_text = input_data["Text"].str.replace("\n", " ")

        results = pd.DataFrame(cleaned_text).apply(lambda x: self.predict(x[0]), result_type="expand", axis="columns")
        results.columns = ["Label", "Score"]
        results.insert(0, "Id", input_data["Id"])

        await ctx.set_output_dataframe("Results", results)

    def get_model_url(self, ctx: mlservicewrapper.core.contexts.ServiceContext):
        return "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
