#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace sim_cuda::benchmark::report {

struct Measurement {
  std::string workload;
  std::size_t environments;
  int substeps;
  bool accuracy_pass;
  double event_min_ms, event_median_ms, event_max_ms;
  double wall_min_ms, wall_median_ms, wall_max_ms;
};

struct Run {
  std::string label;
  std::map<std::string, std::string> compatibility;
  std::vector<Measurement> measurements;
};

namespace detail {

inline constexpr std::size_t max_line_bytes = 65536;
inline constexpr std::size_t max_file_bytes = 16 * 1024 * 1024;
inline constexpr std::size_t max_metadata = 1024;
inline constexpr std::size_t max_measurements = 10000;
using Key = std::tuple<std::string, std::size_t, int>;

[[noreturn]] inline void fail(const std::string &message) {
  throw std::runtime_error("benchmark report: " + message);
}

inline void validate_text(std::string_view value) {
  if (value.empty() || value.size() > max_line_bytes ||
      value.find_first_of("\t\r\n") != std::string_view::npos ||
      value.find('\0') != std::string_view::npos) {
    fail("empty, oversized, or invalid text field");
  }
}

inline void validate_times(double minimum, double median, double maximum) {
  if (!std::isfinite(minimum) || !std::isfinite(median) ||
      !std::isfinite(maximum) || minimum <= 0.0 || minimum > median ||
      median > maximum) {
    fail("timings must be finite, positive, and min <= median <= max");
  }
}

inline std::map<Key, const Measurement *> index(const Run &run) {
  validate_text(run.label);
  if (run.compatibility.empty() || run.compatibility.size() > max_metadata ||
      run.measurements.empty() || run.measurements.size() > max_measurements) {
    fail("missing or excessive metadata or measurements");
  }
  for (const auto &[key, value] : run.compatibility) {
    validate_text(key);
    validate_text(value);
  }
  std::map<Key, const Measurement *> rows;
  for (const auto &measurement : run.measurements) {
    validate_text(measurement.workload);
    if (measurement.environments == 0 || measurement.substeps <= 0) {
      fail("environment count and substeps must be positive");
    }
    validate_times(measurement.event_min_ms, measurement.event_median_ms,
                   measurement.event_max_ms);
    validate_times(measurement.wall_min_ms, measurement.wall_median_ms,
                   measurement.wall_max_ms);
    if (!rows.emplace(Key{measurement.workload, measurement.environments,
                          measurement.substeps},
                      &measurement)
             .second) {
      fail("duplicate workload/count/substep row");
    }
  }
  return rows;
}

inline std::vector<std::string_view> fields(const std::string &line,
                                            std::size_t expected) {
  std::vector<std::string_view> result;
  result.reserve(expected);
  std::string_view remaining = line;
  while (true) {
    const auto separator = remaining.find('\t');
    result.push_back(remaining.substr(0, separator));
    if (result.size() > expected) {
      fail("too many fields");
    }
    if (separator == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(separator + 1);
  }
  if (result.size() != expected) {
    fail("incorrect field count");
  }
  return result;
}

template <typename T> inline T number(std::string_view text) {
  T value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} ||
      end != text.data() + text.size()) {
    fail("invalid numeric field");
  }
  return value;
}

// Counts plus a mandatory terminator detect reports cut off at row boundaries.
inline Run read_report(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("cannot open baseline: " + path);
  }
  std::size_t bytes = 0;
  const auto read_line = [&]() {
    std::string line;
    char c;
    while (input.get(c)) {
      if (++bytes > max_file_bytes) {
        fail("report exceeds size limit");
      }
      if (c == '\n') {
        return line;
      }
      if (line.size() == max_line_bytes || c == '\r' || c == '\0') {
        fail("oversized or invalid line");
      }
      line.push_back(c);
    }
    fail("truncated report or read error");
  };
  if (read_line() != "sim_cuda_physics_benchmark\t1") {
    fail("incompatible report schema");
  }
  Run run;
  auto line = read_line();
  auto parts = fields(line, 2);
  if (parts[0] != "label") {
    fail("missing label record");
  }
  run.label = parts[1];
  line = read_line();
  parts = fields(line, 2);
  if (parts[0] != "metadata") {
    fail("missing metadata count");
  }
  const auto metadata_count = number<std::size_t>(parts[1]);
  if (metadata_count == 0 || metadata_count > max_metadata) {
    fail("invalid metadata count");
  }
  for (std::size_t i = 0; i < metadata_count; ++i) {
    line = read_line();
    parts = fields(line, 3);
    if (parts[0] != "meta" ||
        !run.compatibility.emplace(parts[1], parts[2]).second) {
      fail("invalid or duplicate metadata record");
    }
  }
  line = read_line();
  parts = fields(line, 2);
  if (parts[0] != "measurements") {
    fail("missing measurement count");
  }
  const auto measurement_count = number<std::size_t>(parts[1]);
  if (measurement_count == 0 || measurement_count > max_measurements) {
    fail("invalid measurement count");
  }
  run.measurements.reserve(measurement_count);
  for (std::size_t i = 0; i < measurement_count; ++i) {
    line = read_line();
    parts = fields(line, 11);
    if (parts[0] != "measurement" ||
        (parts[4] != "pass" && parts[4] != "fail")) {
      fail("invalid measurement record or accuracy status");
    }
    run.measurements.push_back(
        {std::string(parts[1]), number<std::size_t>(parts[2]),
         number<int>(parts[3]), parts[4] == "pass", number<double>(parts[5]),
         number<double>(parts[6]), number<double>(parts[7]),
         number<double>(parts[8]), number<double>(parts[9]),
         number<double>(parts[10])});
  }
  if (read_line() != "end") {
    fail("missing report terminator");
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
    fail("trailing data or read error");
  }
  index(run);
  return run;
}

inline void timing_delta(std::ostream &out, std::string_view name,
                         double baseline_min, double baseline_median,
                         double baseline_max, double current_min,
                         double current_median, double current_max) {
  const long double delta =
      (static_cast<long double>(current_median) / baseline_median - 1.0L) *
      100.0L;
  const bool overlap =
      baseline_min <= current_max && current_min <= baseline_max;
  out << "  " << name << " median delta=" << std::showpos << delta
      << std::noshowpos
      << "% (current vs baseline); median ms=" << baseline_median << " -> "
      << current_median << "; min/max ms baseline=[" << baseline_min << ", "
      << baseline_max << "], current=[" << current_min << ", " << current_max
      << "]; sample ranges " << (overlap ? "overlap" : "do not overlap")
      << '\n';
}

} // namespace detail

inline void write_report(const std::string &path, const Run &run) {
  const auto rows = detail::index(run);
  std::ostringstream content;
  content.imbue(std::locale::classic());
  content << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "sim_cuda_physics_benchmark\t1\nlabel\t" << run.label
          << "\nmetadata\t" << run.compatibility.size() << '\n';
  for (const auto &[key, value] : run.compatibility) {
    content << "meta\t" << key << '\t' << value << '\n';
  }
  content << "measurements\t" << rows.size() << '\n';
  for (const auto &[key, row] : rows) {
    content << "measurement\t" << row->workload << '\t' << row->environments
            << '\t' << row->substeps << '\t'
            << (row->accuracy_pass ? "pass" : "fail") << '\t'
            << row->event_min_ms << '\t' << row->event_median_ms << '\t'
            << row->event_max_ms << '\t' << row->wall_min_ms << '\t'
            << row->wall_median_ms << '\t' << row->wall_max_ms << '\n';
  }
  content << "end\n";
  const auto serialized = content.str();
  if (serialized.size() > detail::max_file_bytes) {
    detail::fail("report exceeds size limit");
  }
  std::size_t start = 0;
  while (start < serialized.size()) {
    const auto end = serialized.find('\n', start);
    if (end - start > detail::max_line_bytes) {
      detail::fail("report line exceeds size limit");
    }
    start = end + 1;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    detail::fail("cannot open output: " + path);
  }
  output << serialized;
  output.close();
  if (!output) {
    detail::fail("cannot write output: " + path);
  }
}

inline void compare_report(const std::string &path, const Run &current,
                           std::ostream &out) {
  const auto baseline = detail::read_report(path);
  const auto baseline_rows = detail::index(baseline);
  const auto current_rows = detail::index(current);
  if (baseline.compatibility != current.compatibility) {
    detail::fail("incompatible hardware/configuration/scenario metadata");
  }
  if (baseline_rows.size() != current_rows.size()) {
    detail::fail("incompatible workload/count/substep key sets");
  }
  for (const auto &[key, row] : baseline_rows) {
    if (!current_rows.contains(key)) {
      detail::fail("incompatible workload/count/substep key sets");
    }
  }
  // Validate the complete run before emitting any comparison claims.
  std::ostringstream result;
  result.imbue(std::locale::classic());
  result << std::setprecision(6)
         << "Benchmark comparison: baseline=\"" << baseline.label
         << "\", current=\"" << current.label << "\"\n"
         << "Caution: clocks and system load are uncontrolled; repeat "
            "measurements. "
            "Deltas and sample min/max ranges are descriptive, not confidence "
            "intervals or a performance pass/fail threshold. Negative deltas "
            "mean lower median time.\n";
  for (const auto &[key, old_row] : baseline_rows) {
    const auto &row = *current_rows.at(key);
    result << row.workload << " environments=" << row.environments
           << " substeps=" << row.substeps << '\n';
    if (!old_row->accuracy_pass || !row.accuracy_pass) {
      result << "  INELIGIBLE: accuracy baseline="
             << (old_row->accuracy_pass ? "pass" : "fail")
             << ", current=" << (row.accuracy_pass ? "pass" : "fail")
             << "; no performance percentage reported.\n";
      continue;
    }
    detail::timing_delta(result, "event", old_row->event_min_ms,
                         old_row->event_median_ms, old_row->event_max_ms,
                         row.event_min_ms, row.event_median_ms,
                         row.event_max_ms);
    detail::timing_delta(result, "wall", old_row->wall_min_ms,
                         old_row->wall_median_ms, old_row->wall_max_ms,
                         row.wall_min_ms, row.wall_median_ms, row.wall_max_ms);
  }
  out << result.str();
  if (!out) {
    detail::fail("cannot write comparison output");
  }
}

} // namespace sim_cuda::benchmark::report
