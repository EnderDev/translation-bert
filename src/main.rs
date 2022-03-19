use rust_bert::pipelines::translation::{ Language, TranslationModelBuilder };
use std::time::{ SystemTime, UNIX_EPOCH };

fn translate(from: Language, to: Language, text: &str) -> String {
    let start = SystemTime::now();
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start_ms = since_the_epoch.as_millis();

    let model = TranslationModelBuilder::new()
        .with_source_languages(vec![from])
        .with_target_languages(vec![to])
        .create_model()
        .unwrap();

    let outputs = model
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

fn main() {
    let a = translate(
        Language::English, 
        Language::Spanish, 
        "Hello, my name is Bingus. I love Dot Browser."
    );
    println!("{}", a);

    let b = translate(
        Language::Spanish, 
        Language::English, 
        &a
    );
    println!("{}", b);
}
