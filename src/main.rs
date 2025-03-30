use chrono::{DateTime, TimeZone, Utc};
use mpi::{self, collective::Root, environment::Universe, traits::*};
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, Value};
use std::cmp::{min, Ordering};
use std::collections::{BinaryHeap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use clap::{Arg, Command};
use std::os::unix::fs::MetadataExt;

// -----------------------------------
// Config module - from config.py
// -----------------------------------
struct Config {
    data_dir: String,
    output_dir: String,
    log_dir: String,
    preprocessed_data_prefix: String,
    input_file_pattern: String,
    default_input_file: String,
    cloud_storage_bucket: Option<String>,
    cloud_input_path: Option<String>,
    cloud_output_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            data_dir: "/Users/manishghoshal/Desktop/UniMelb/Semester 1/CCC/CCC_A1/data/mastodon-16m.ndjson".to_string(),
            output_dir: "/Users/manishghoshal/Desktop/UniMelb/Semester 1/CCC/CCC_A1/output".to_string(),
            log_dir: "./logs".to_string(),
            preprocessed_data_prefix: "preprocessed_".to_string(),
            input_file_pattern: "*.ndjson".to_string(),
            default_input_file: "mastodon_data.ndjson".to_string(),
            cloud_storage_bucket: None,
            cloud_input_path: None,
            cloud_output_path: None,
        }
    }
}

impl Config {
    fn load(env: Option<&str>, custom_config: Option<HashMap<String, String>>) -> Self {
        let mut config = Config::default();
        
        if let Some(env_name) = env {
            match env_name {
                "development" => {
                    config.data_dir = "./dev_data".to_string();
                    config.output_dir = "./dev_output".to_string();
                },
                "production" => {
                    config.data_dir = "/opt/mastodon/data".to_string();
                    config.output_dir = "/opt/mastodon/output".to_string();
                },
                "testing" => {
                    config.data_dir = "./test_data".to_string();
                    config.output_dir = "./test_output".to_string();
                },
                _ => {}
            }
        }
        
        if let Some(custom) = custom_config {
            if let Some(data_dir) = custom.get("data_dir") {
                config.data_dir = data_dir.clone();
            }
            if let Some(output_dir) = custom.get("output_dir") {
                config.output_dir = output_dir.clone();
            }
            // Add other fields as needed
        }
        
        config
    }
    
    fn resolve_path(&self, base_dir: &str, filename: &str) -> PathBuf {
        let dir_path = match base_dir {
            "data_dir" => &self.data_dir,
            "output_dir" => &self.output_dir,
            "log_dir" => &self.log_dir,
            _ => base_dir,
        };
        
        let path = Path::new(dir_path);
        if !path.exists() {
            fs::create_dir_all(path).expect("Failed to create directory");
        }
        
        if filename.is_empty() {
            path.to_path_buf()
        } else {
            path.join(filename)
        }
    }
}

// -----------------------------------
// MastodonData struct - from MastodonData.py
// -----------------------------------
#[derive(Debug, Deserialize, Serialize, Clone)]
struct MastodonData {
    created_at: Option<String>,
    user_id: Option<String>,
    username: Option<String>,
    sentiment: Option<f64>,
}

impl MastodonData {
    fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        let data: Value = from_str(json_str)?;
        
        // Extract fields with proper error handling
        let created_at = data.get("created_at").and_then(|v| v.as_str()).map(String::from);
        let user_id = data.get("account").and_then(|a| a.get("id")).and_then(|v| v.as_str()).map(String::from);
        let username = data.get("account").and_then(|a| a.get("username")).and_then(|v| v.as_str()).map(String::from);
        
        // Extract sentiment (assuming it's directly in the JSON or calculated)
        // In the original, this might be calculated rather than directly present
        let sentiment = data.get("sentiment").and_then(|v| v.as_f64());
        
        Ok(MastodonData {
            created_at,
            user_id,
            username,
            sentiment,
        })
    }
}

// -----------------------------------
// Utility functions - from util.py
// -----------------------------------
const SEPARATOR: &str = "==================================================";

fn preprocess_data(data: &str) -> Option<String> {
    let trimmed = data.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn processing_data(
    preprocessed_line: &str,
    hour_sentiment_dict: &mut HashMap<String, f64>,
    user_sentiment_dict: &mut HashMap<String, (String, f64)>,
) {
    match MastodonData::from_json_str(preprocessed_line) {
        Ok(mastodon_data) => {
            // Skip entries without required fields
            if mastodon_data.created_at.is_none() || mastodon_data.sentiment.is_none() {
                return;
            }
            
            // Process date
            if let Some(created_at) = &mastodon_data.created_at {
                let created_at = created_at.replace('Z', "+00:00");
                if let Ok(created_datetime) = DateTime::parse_from_rfc3339(&created_at) {
                    let hour_key = format!("{}", created_datetime.format("%Y-%m-%d %H"));
                    
                    if let Some(sentiment) = mastodon_data.sentiment {
                        *hour_sentiment_dict.entry(hour_key).or_insert(0.0) += sentiment;
                    }
                }
            }
            
            // Process user sentiment
            if let (Some(user_id), Some(username), Some(sentiment)) = 
               (mastodon_data.user_id, mastodon_data.username, mastodon_data.sentiment) {
                let entry = user_sentiment_dict.entry(user_id).or_insert((username.clone(), 0.0));
                entry.1 += sentiment;
            }
        },
        Err(_) => {
            // Skip entries that can't be processed
        }
    }
}

fn format_hour_range(hour_str: &str) -> String {
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&format!("{hour_str}:00:00"), "%Y-%m-%d %H:%M:%S") {
        let end_hour = dt.hour() + 1;
        format!("{} to {} {:02}:00", 
                dt.format("%Y-%m-%d %H:00"), 
                dt.format("%Y-%m-%d"), 
                end_hour)
    } else {
        hour_str.to_string()
    }
}

fn dump_happiest_hours(happy_hours: &[(String, f64)], output_dir: &Path) {
    println!("{}", SEPARATOR);
    println!("Top Happiest Hours");
    println!("{}", SEPARATOR);
    
    let mut output = Vec::new();
    for (i, (hour, score)) in happy_hours.iter().enumerate() {
        let formatted_hour = format_hour_range(hour);
        let line = format!("{}. {} with sentiment +{}", i + 1, formatted_hour, score);
        println!("{}", line);
        output.push(line);
    }
    println!();
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    // Write to file
    let output_file = output_dir.join("happiest_hours.txt");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_file)
        .expect("Failed to open happiest_hours.txt for writing");
    
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Top Happiest Hours").expect("Failed to write to file");
    writeln!(writer, "{}", SEPARATOR).expect("Failed to write to file");
    
    for line in output {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    }
}

fn dump_saddest_hours(sad_hours: &[(String, f64)], output_dir: &Path) {
    println!("{}", SEPARATOR);
    println!("Top Saddest Hours");
    println!("{}", SEPARATOR);
    
    let mut output = Vec::new();
    for (i, (hour, score)) in sad_hours.iter().enumerate() {
        let formatted_hour = format_hour_range(hour);
        let line = format!("{}. {} with sentiment {}", i + 1, formatted_hour, score);
        println!("{}", line);
        output.push(line);
    }
    println!();
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    // Write to file
    let output_file = output_dir.join("saddest_hours.txt");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_file)
        .expect("Failed to open saddest_hours.txt for writing");
    
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Top Saddest Hours").expect("Failed to write to file");
    writeln!(writer, "{}", SEPARATOR).expect("Failed to write to file");
    
    for line in output {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    }
}

fn dump_happiest_users(happy_users: &[(String, (String, f64))], output_dir: &Path) {
    println!("{}", SEPARATOR);
    println!("Top Happiest Users");
    println!("{}", SEPARATOR);
    
    let mut output = Vec::new();
    for (i, (user_id, (username, score))) in happy_users.iter().enumerate() {
        let line = format!("{}. {} (ID: {}) with total sentiment +{}", i + 1, username, user_id, score);
        println!("{}", line);
        output.push(line);
    }
    println!();
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    // Write to file
    let output_file = output_dir.join("happiest_users.txt");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_file)
        .expect("Failed to open happiest_users.txt for writing");
    
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Top Happiest Users").expect("Failed to write to file");
    writeln!(writer, "{}", SEPARATOR).expect("Failed to write to file");
    
    for line in output {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    }
}

fn dump_saddest_users(sad_users: &[(String, (String, f64))], output_dir: &Path) {
    println!("{}", SEPARATOR);
    println!("Top Saddest Users");
    println!("{}", SEPARATOR);
    
    let mut output = Vec::new();
    for (i, (user_id, (username, score))) in sad_users.iter().enumerate() {
        let line = format!("{}. {} (ID: {}) with total sentiment {}", i + 1, username, user_id, score);
        println!("{}", line);
        output.push(line);
    }
    println!();
    
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).expect("Failed to create output directory");
    
    // Write to file
    let output_file = output_dir.join("saddest_users.txt");
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_file)
        .expect("Failed to open saddest_users.txt for writing");
    
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Top Saddest Users").expect("Failed to write to file");
    writeln!(writer, "{}", SEPARATOR).expect("Failed to write to file");
    
    for line in output {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    }
}

fn dump_time(comm_rank: i32, title: &str, time_period: f64) {
    println!("{}", SEPARATOR);
    println!("Processor #{} completed {} in {:.2} seconds", comm_rank, title, time_period);
    println!("{}", SEPARATOR);
}

fn dump_num_processor(comm_size: usize) {
    println!("{}", SEPARATOR.repeat(2));
    println!("Running with {} processors", comm_size);
    println!("{}", SEPARATOR.repeat(2));
    println!();
}

// Helper struct to manage topN heap operations
#[derive(Debug, Clone, PartialEq)]
struct SentimentItem<T> {
    key: T,
    value: f64,
}

impl<T: Eq> Eq for SentimentItem<T> {}

impl<T: PartialEq> PartialOrd for SentimentItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Eq> Ord for SentimentItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.partial_cmp(&other.value).unwrap_or(Ordering::Equal)
    }
}

// Helper struct for user sentiment heaps
#[derive(Debug, Clone, PartialEq)]
struct UserSentimentItem {
    user_id: String,
    username: String,
    sentiment: f64,
}

impl Eq for UserSentimentItem {}

impl PartialOrd for UserSentimentItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sentiment.partial_cmp(&other.sentiment)
    }
}

impl Ord for UserSentimentItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sentiment.partial_cmp(&other.sentiment).unwrap_or(Ordering::Equal)
    }
}

// -----------------------------------
// Main processing functions - from main.py
// -----------------------------------
fn process_chunk_memory_mapped(
    input_file: &str,
    local_start: u64,
    local_end: u64,
    max_buffer_size: usize,
) -> (HashMap<String, f64>, HashMap<String, (String, f64)>, usize) {
    let mut local_hour_sentiment = HashMap::new();
    let mut local_user_sentiment = HashMap::new();
    let mut lines_processed = 0;
    
    // Open file with memory mapping
    let file = File::open(input_file).expect("Failed to open input file");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to map file") };
    
    // Set initial position
    let mut position = local_start as usize;
    
    // Find the next newline if not at start
    if position > 0 {
        if let Some(pos) = mmap[position..].iter().position(|&b| b == b'\n') {
            position += pos + 1; // Move past the newline
        } else {
            // No more newlines
            return (local_hour_sentiment, local_user_sentiment, lines_processed);
        }
    }
    
    // Process the file in smaller segments to avoid OOM
    let mut current_pos = position;
    let mut segment_start = current_pos;
    
    while current_pos < local_end as usize && current_pos < mmap.len() {
        // Calculate the end of the current segment
        let segment_end = min(segment_start + max_buffer_size, local_end as usize);
        let segment_end = min(segment_end, mmap.len());
        
        // Adjust segment_end to the next newline if not at the end of our chunk
        let mut adjusted_segment_end = segment_end;
        if segment_end < local_end as usize && segment_end < mmap.len() {
            if let Some(pos) = mmap[segment_end..].iter().position(|&b| b == b'\n') {
                adjusted_segment_end = segment_end + pos + 1;
            }
        }
        
        // Process lines in the current segment
        while current_pos < adjusted_segment_end && current_pos < mmap.len() {
            let next_newline = if let Some(pos) = mmap[current_pos..adjusted_segment_end]
                .iter()
                .position(|&b| b == b'\n') 
            {
                current_pos + pos
            } else {
                // No newline found; move current_pos to segment_end
                adjusted_segment_end
            };
            
            if next_newline > current_pos {
                // Try to decode as UTF-8
                if let Ok(line) = std::str::from_utf8(&mmap[current_pos..next_newline]) {
                    // Use the preprocess_data function
                    if let Some(pre_line) = preprocess_data(line) {
                        // Use the processing_data function
                        processing_data(&pre_line, &mut local_hour_sentiment, &mut local_user_sentiment);
                        lines_processed += 1;
                    }
                }
            }
            
            // Move past the newline
            current_pos = next_newline + 1;
            if current_pos >= mmap.len() {
                break;
            }
        }
        
        segment_start = current_pos;
    }
    
    (local_hour_sentiment, local_user_sentiment, lines_processed)
}

fn merge_hour_dicts(dicts_list: Vec<HashMap<String, f64>>) -> HashMap<String, f64> {
    let mut merged = HashMap::new();
    for dict in dicts_list {
        for (hour, sentiment) in dict {
            *merged.entry(hour).or_insert(0.0) += sentiment;
        }
    }
    merged
}

fn merge_user_dicts(dicts_list: Vec<HashMap<String, (String, f64)>>) -> HashMap<String, (String, f64)> {
    let mut merged = HashMap::new();
    for dict in dicts_list {
        for (uid, (username, sentiment)) in dict {
            let entry = merged.entry(uid).or_insert((username.clone(), 0.0));
            entry.1 += sentiment;
        }
    }
    merged
}

fn setup_mpi_file_boundaries(input_file: &str, rank: usize, size: usize) -> (u64, u64, u64) {
    let metadata = fs::metadata(input_file).expect("Failed to get file metadata");
    let file_size = metadata.size();
    
    let chunk_size = file_size / size as u64;
    let local_start = rank as u64 * chunk_size;
    let local_end = if rank == size - 1 {
        file_size
    } else {
        (rank as u64 + 1) * chunk_size
    };
    
    (local_start, local_end, file_size)
}

// Function to find top-n items by value
fn top_n_by_value<T: Clone>(map: &HashMap<T, f64>, n: usize, largest: bool) -> Vec<(T, f64)> 
where T: Ord + Clone 
{
    let mut heap = BinaryHeap::with_capacity(n + 1);
    
    for (key, &value) in map {
        let item = if largest {
            SentimentItem { key: key.clone(), value }
        } else {
            SentimentItem { key: key.clone(), value: -value }
        };
        
        heap.push(item);
        if heap.len() > n {
            heap.pop();
        }
    }
    
    let mut result = Vec::with_capacity(n);
    while let Some(item) = heap.pop() {
        let value = if largest { item.value } else { -item.value };
        result.push((item.key, value));
    }
    
    result.reverse();
    result
}

// Function to find top-n users by sentiment
fn top_n_users(map: &HashMap<String, (String, f64)>, n: usize, largest: bool) -> Vec<(String, (String, f64))> {
    let mut heap = BinaryHeap::with_capacity(n + 1);
    
    for (user_id, (username, sentiment)) in map {
        let item = UserSentimentItem {
            user_id: user_id.clone(),
            username: username.clone(),
            sentiment: if largest { *sentiment } else { -sentiment },
        };
        
        heap.push(item);
        if heap.len() > n {
            heap.pop();
        }
    }
    
    let mut result = Vec::with_capacity(n);
    while let Some(item) = heap.pop() {
        let sentiment = if largest { item.sentiment } else { -item.sentiment };
        result.push((item.user_id, (item.username, sentiment)));
    }
    
    result.reverse();
    result
}

// -----------------------------------
// Main function - entry point
// -----------------------------------
fn main() -> io::Result<()> {
    // Initialize MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;
    
    let start_time = Instant::now();
    
    // Parse command line arguments
    let matches = Command::new("Mastodon Data Analytics")
        .arg(Arg::new("data")
            .short('d')
            .long("data")
            .value_name("FILE")
            .help("Path to Mastodon NDJSON file")
            .required(true))
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .value_name("DIR")
            .help("Output directory for results")
            .required(false))
        .arg(Arg::new("buffer-size")
            .long("buffer-size")
            .value_name("SIZE")
            .help("Buffer size in MB for processing chunks (default: 100)")
            .default_value("100"))
        .get_matches();
    
    let data_file = matches.get_one::<String>("data").unwrap();
    
    // Initialize config
    let config = Config::default();
    
    // Get output directory from command line or config
    let output_dir = if let Some(output) = matches.get_one::<String>("output") {
        PathBuf::from(output)
    } else {
        PathBuf::from(&config.output_dir)
    };
    
    // Buffer size
    let buffer_size: usize = matches.get_one::<String>("buffer-size")
        .unwrap()
        .parse()
        .unwrap_or(100);
    let buffer_size_bytes = buffer_size * 1024 * 1024;
    
    if rank == 0 {
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");
        dump_num_processor(size);
    }
    
    // Set up file boundaries for MPI
    let (local_start, local_end, _) = setup_mpi_file_boundaries(data_file, rank, size);
    
    // Process the data
    let processing_start = Instant::now();
    let (local_hour_sentiment, local_user_sentiment, local_lines_processed) = 
        process_chunk_memory_mapped(data_file, local_start, local_end, buffer_size_bytes);
    let processing_time = processing_start.elapsed().as_secs_f64();
    
    dump_time(rank as i32, "data processing", processing_time);
    
    // Wait for all processes
    world.barrier();
    
    // Gather results from all processes
    let gathering_start = Instant::now();
    
    // Serialize and gather hour sentiment dictionaries
    let mut all_hour_dicts = Vec::new();
    if rank == 0 {
        all_hour_dicts.push(local_hour_sentiment.clone());
        for i in 1..size {
            let from_rank = i as i32;
            let hour_dict: HashMap<String, f64> = world.process_at_rank(from_rank).receive().0;
            all_hour_dicts.push(hour_dict);
        }
    } else {
        world.process_at_rank(0).send(&local_hour_sentiment);
    }
    
    // Serialize and gather user sentiment dictionaries
    let mut all_user_dicts = Vec::new();
    if rank == 0 {
        all_user_dicts.push(local_user_sentiment.clone());
        for i in 1..size {
            let from_rank = i as i32;
            let user_dict: HashMap<String, (String, f64)> = world.process_at_rank(from_rank).receive().0;
            all_user_dicts.push(user_dict);
        }
    } else {
        world.process_at_rank(0).send(&local_user_sentiment);
    }
    
    // Gather total processed lines
    let mut total_lines = local_lines_processed;
    if rank == 0 {
        for i in 1..size {
            let from_rank = i as i32;
            let lines: usize = world.process_at_rank(from_rank).receive().0;
            total_lines += lines;
        }
    } else {
        world.process_at_rank(0).send(&local_lines_processed);
    }
    
    let gathering_time = gathering_start.elapsed().as_secs_f64();
    
    // Process the gathered data on rank 0
    if rank == 0 {
        let merging_start = Instant::now();
        
        // Merge dictionaries
        let global_hour_sentiment = merge_hour_dicts(all_hour_dicts);
        let global_user_sentiment = merge_user_dicts(all_user_dicts);
        
        let merging_time = merging_start.elapsed().as_secs_f64();
        
        // Find top N items
        let top_n = 5;
        let happiest_hours = top_n_by_value(&global_hour_sentiment, top_n, true);
        let saddest_hours = top_n_by_value(&global_hour_sentiment, top_n, false);
        let happiest_users = top_n_users(&global_user_sentiment, top_n, true);
        let saddest_users = top_n_users(&global_user_sentiment, top_n, false);
        
        // Output results
        dump_happiest_hours(&happiest_hours, &output_dir);
        dump_saddest_hours(&saddest_hours, &output_dir);
        dump_happiest_users(&happiest_users, &output_dir);
        dump_saddest_users(&saddest_users, &output_dir);
        
        let total_time = start_time.elapsed().as_secs_f64();
        println!("Total processing time: {:.2} seconds", total_time);
        println!("Total lines processed: {}", total_lines);
        println!("Program runs in {:.2} seconds", total_time);
    }
    
    Ok(())
}