package radios.restservice;

import java.io.IOException;
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
	public void receiveIQ(@RequestParam String name) {
        System.out.print(name);
		boolean isWindows = System.getProperty("os.name").toLowerCase().startsWith("windows");
		try{
		Process process;
		if (isWindows) {
			process = Runtime.getRuntime().exec("echo hello");
		} else {
			process = Runtime.getRuntime().exec("echo hello");
		}		// return new Greeting(counter.incrementAndGet(), String.format(template, name));
		}
		catch(IOException e){
			System.err.println("error");
		}
	}
}