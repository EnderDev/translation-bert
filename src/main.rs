use rust_bert::{pipelines::{translation::{ Language, TranslationConfig, TranslationModel }, common::ModelType}, m2m_100::{M2M100ModelResources, M2M100ConfigResources, M2M100TargetLanguages, M2M100SourceLanguages, M2M100VocabResources, M2M100MergesResources}, resources::RemoteResource};
use tch::{Device};
use std::{time::{ SystemTime, UNIX_EPOCH }};

#[derive()]
struct Translation {
    model: TranslationModel,
}
impl Translation {
    fn translate(&self, from: Language, to: Language, text: &str) -> String {
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start_ms = since_the_epoch.as_millis();
    
        let outputs = self.model
            .translate(&[text], from, to)
            .unwrap();
    
        let output = format!("{}", &outputs[0].trim());
    
        let end = SystemTime::now();
        let since_the_epoch = end
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let end_ms = since_the_epoch.as_millis();
    
        println!("Translation took {}ms", end_ms - start_ms);
    
        return output.to_string()
    }

    pub fn new() -> Self {
        println!("Loading M2M100 models...");

        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start_ms = since_the_epoch.as_millis();
        
        let model_resource = RemoteResource::from_pretrained(M2M100ModelResources::M2M100_418M);
        let config_resource = RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_418M);
        let vocab_resource = RemoteResource::from_pretrained(M2M100VocabResources::M2M100_418M);
        let merges_resource = RemoteResource::from_pretrained(M2M100MergesResources::M2M100_418M);

        let source_languages = M2M100SourceLanguages::M2M100_418M;
        let target_languages = M2M100TargetLanguages::M2M100_418M;

        match Device::is_cuda(Device::cuda_if_available()) {
            true => println!("Using CUDA for computation."),
            _ => println!("Downgrading to CPU for computation."),
        }

        let translation_config = TranslationConfig {
            model_type: ModelType::M2M100,
            model_resource: rust_bert::resources::Resource::Remote(model_resource),
            config_resource: rust_bert::resources::Resource::Remote(config_resource),
            vocab_resource: rust_bert::resources::Resource::Remote(vocab_resource),
            merges_resource: rust_bert::resources::Resource::Remote(merges_resource),
            source_languages: source_languages.as_ref().iter().cloned().collect(),
            target_languages: target_languages.as_ref().iter().cloned().collect(),
            min_length: 0,
            max_length: 512,
            do_sample: false,
            early_stopping: true,
            num_beams: 4,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            device: Device::cuda_if_available(),
            num_beam_groups: None,
            diversity_penalty: None,
        };

        let model = TranslationModel::new(translation_config).unwrap();

        let end = SystemTime::now();
        let since_the_epoch = end
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let end_ms = since_the_epoch.as_millis();
    
        println!("Loaded models in {}ms", end_ms - start_ms);

        Self {
            model
        }
    }
}

fn main() {
    let engine = Translation::new();

    let a = engine.translate(
        Language::English, 
        Language::Spanish, 
        "Hello, my name is Bingus. I love Dot Browser."
    );
    println!("{}", a);

    let b = engine.translate(
        Language::Spanish, 
        Language::English, 
        &a
    );
    println!("{}", b);
}
