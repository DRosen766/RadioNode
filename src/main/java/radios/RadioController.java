package radios.restservice;

import java.util.concurrent.atomic.AtomicLong;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RadioController {

	private static final String template = "Hello, %s!";
	private final AtomicLong counter = new AtomicLong();

	@PostMapping("/receiveiq")
	public void receiveIQ(@RequestParam(value = "name", defaultValue = "World") String name) {
        System.out.print("Hello world!!");
		// return new Greeting(counter.incrementAndGet(), String.format(template, name));
	}
}