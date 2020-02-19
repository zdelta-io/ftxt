extern crate serde;
#[macro_use]
extern crate serde_derive;
use std::env;
use std::vec::Vec;
use std::error::Error;
use std::io::Write;
use std::fs::File;
use std::collections::HashSet;

use csv;
use rand::thread_rng;
use rand::seq::SliceRandom;
use fasttext::{FastText, Args, ModelName, LossName, Prediction};
use stopwords::{Spark, Language, Stopwords};
use itertools::Itertools;
use vtext::tokenize::*;
use rust_stemmers::{Algorithm, Stemmer};

const TRAIN_FILE: &str = "./dataset/train.data";
const TEST_FILE: &str = "./dataset/test.data";
const MODEL: &str = "./dataset/model.bin";

#[derive(Debug, Deserialize, Clone)]
struct SpookyAuthor {
    id: String,
    text: String,
    author: String,
}

impl SpookyAuthor {
    fn extract_labels(&self) -> String {
        match self.author.as_str() {
            "EAP" => "__label__EAP".to_owned(),
            "HPL" => "__label__HPL".to_owned(),
            "MWS" => "__label__MWS".to_owned(),
            l => {
                panic!(
                    "Not able to parse the target string. \
                Got something else: {:?}",
                    l
                )
            }
        }
    }

    pub fn extract_tokens(&self) -> String {
        let text_lowercase = self.text.to_lowercase();

        // Tokenize 分词
        let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
        let tokens: Vec<&str> = tokenizer.tokenize(text_lowercase.as_str()).collect();

        // Stemming
        let stemmer = Stemmer::create(Algorithm::English);
        let tokens: Vec<String> = tokens
            .iter()
            .map(|tok| stemmer.stem(tok).into_owned())
            .collect();
        let mut tokens: Vec<&str> = tokens.iter().map(|tok| tok.as_str()).collect();

        // 移除停词(Stop Words)
        let stop_words: HashSet<_> = Spark::stopwords(Language::English)
            .unwrap()
            .iter()
            .collect();
        tokens.retain(|tok| !stop_words.contains(tok));

        tokens.iter().join(" ")
    }
}

fn main() -> Result<(), Box<Error>> {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];

    let data = extract_data(&path).unwrap();
    let (test_data, train_data) = data_splitter(&data, 0.2);
    write_training_data(&train_data.to_owned(), TRAIN_FILE)?;
    write_test_data(&test_data.to_owned(), TEST_FILE)?;

    let mut ftxt_model = FastText::new();
    model_training(TRAIN_FILE, &mut ftxt_model);
    check_accuracy(&test_data, &ftxt_model);

    ftxt_model.save_model(MODEL)?;
    Ok(())
}

fn extract_data(path: &String) -> Result<Vec<SpookyAuthor>, Box<Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut data = Vec::new();
    for record in rdr.deserialize() {
        let rec: SpookyAuthor = record?;
        data.push(rec);
    }
    data.shuffle(&mut thread_rng());
    Ok(data)
}

fn data_splitter(data: &[SpookyAuthor], test_size: f32) -> (Vec<SpookyAuthor>, Vec<SpookyAuthor>) {
    let test_size: f32 = data.len() as f32 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    (test_data.to_vec(), train_data.to_vec())
}

fn write_training_data(train_data: &[SpookyAuthor], file_name: &str) -> Result<(), Box<Error>> {
    let mut file = File::create(file_name)?;
    for item in train_data {
        writeln!(file, "{} {}", item.extract_labels(), item.extract_tokens())?;
    }
    Ok(())
}

fn write_test_data(test_data: &[SpookyAuthor], file_name: &str) -> Result<(), Box<Error>> {
    let mut file = File::create(file_name)?;
    for item in test_data {
        writeln!(file, "{}", item.extract_tokens())?;
    }
    Ok(())
}

fn model_training(train_file: &str, ftxt_model: &mut FastText) {
    let mut args = Args::new();
    args.set_input(train_file);
    args.set_model(ModelName::SUP);
    args.set_loss(LossName::SOFTMAX);
    ftxt_model.train(&args).unwrap();
}

fn check_accuracy(data: &[SpookyAuthor], model: &FastText) {
    let total_hits = data.len();

    let mut predictions = vec![];
    let mut labels = vec![];
    for td in data {
        let label = td.extract_labels();
        let pred: Result<Vec<Prediction>, String> = model.predict(td.text.as_str(), 1, 0.0);
        predictions.push(pred.unwrap());
        labels.push(label);
    }

    let mut hits = 0;
    let mut correct_hits = 0;
    for (pred, lbl) in predictions.iter().zip(labels) {
        let pred = &pred[0];
        if pred.clone().label == lbl {
            correct_hits += 1;
        }
        hits += 1;
    }
    println!(
        "accuracy={} ({}/{}) correct",
        correct_hits as f32 / hits as f32,
        correct_hits,
        total_hits
    );
}
